// QTIP Gaussian Q8-table race kernels.
//
// Same numerical contract as the physical `QtipGaussianPhysicalQ8*A8Direct*`
// kernels in qtip_gaussian_cached.metal (signed-A8 activations, signed INT8
// 65,536-entry Gaussian table, int32 MXU accumulation, identical epilogue), but
// with a different execution structure:
//
//   1. activations are consumed by the MXU as a device tensor operand instead of
//      being staged through a per-lane register fragment (B32/B64);
//   2. V4 restarted-state extraction is branchless;
//   3. software pipelining: table gathers for chunk c+PREFETCH are issued
//      before the MXU work of chunk c;
//   4. B16 kernels (two row fragments, register activation fragment) with the
//      same pipelining, writing batch-rows output directly for `active_batch`
//      tokens (no transpose pass).
//
// V2 streams are the loader-repacked layout, decoded with the same fixture
// extraction the accepted physical V2 kernels use.
//
// Diagnostic variants (DIAG_*) are speed-only probes and are not numerically
// valid decoders.

#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "common/fragment.h"
#include "common/mxu_fragment/ops.h"

using namespace metal;

// ---------------------------------------------------------------------------
// State extraction
// ---------------------------------------------------------------------------

// V2, k2 (4-bit transitions), repacked stream: byte = column / 4.
static inline ushort2 qtip_race_state_pair_k2(
    device const uchar* row_codes,
    uint byte) {
  const uint byte0 = uint(row_codes[byte]);
  const uint byte1 = uint(row_codes[byte + 1u]);
  const uint byte2 = uint(row_codes[byte + 2u]);
  return ushort2(
      ushort((byte0 << 8u) | byte1),
      ushort((byte0 << 12u) | (byte1 << 4u) | (byte2 >> 4u)));
}

// V2, k3 (6-bit transitions), repacked stream: group = column / 2.
static inline ushort2 qtip_race_state_pair_k3(
    device const uchar* row_codes,
    uint group) {
  const uint bit = group * 6u;
  const uint byte = bit >> 3u;
  const uint shift = bit & 7u;
  const uint window =
      (uint(row_codes[byte]) << 24u) |
      (uint(row_codes[byte + 1u]) << 16u) |
      (uint(row_codes[byte + 2u]) << 8u) |
      uint(row_codes[byte + 3u]);
  return ushort2(
      ushort((window >> (16u - shift)) & 0xFFFFu),
      ushort((window >> (10u - shift)) & 0xFFFFu));
}

template <uint TRANSITION_BITS>
static inline ushort2 qtip_race_state_pair_v2(
    device const uchar* row_codes,
    uint column) {
  if constexpr (TRANSITION_BITS == 4u) {
    return qtip_race_state_pair_k2(row_codes, column >> 2u);
  } else {
    return qtip_race_state_pair_k3(row_codes, column >> 1u);
  }
}

// Packed per-byte negation (SWAR): -b = ~b + 1 per byte without cross-byte carries. Apple GPUs emulate
// 8-bit vector arithmetic through 16-bit conversions, so this is far cheaper than negating a char4.
static inline uint qtip_race_negate_bytes(uint x, uint byte_mask) {
  const uint t = x ^ byte_mask;                                   // ~x on the selected bytes
  const uint inc = ((t & (0x7F7F7F7Fu & byte_mask)) + (0x01010101u & byte_mask)) ^ (t & (0x80808080u & byte_mask));
  return (x & ~byte_mask) | (inc & byte_mask);
}
static inline char4 qtip_race_flip_all(char4 v, bool flip) {
  const uint x = as_type<uint>(v);
  return as_type<char4>(flip ? qtip_race_negate_bytes(x, 0xFFFFFFFFu) : x);
}
static inline char4 qtip_race_flip_lo2(char4 v, bool flip) {
  const uint x = as_type<uint>(v);
  return as_type<char4>(flip ? qtip_race_negate_bytes(x, 0x0000FFFFu) : x);
}
static inline char4 qtip_race_flip_bits(char4 v, ushort h) {
  // component c negated when bit c of h is set
  const uint x = as_type<uint>(v);
  const uint mask = ((h & 1u) ? 0x000000FFu : 0u) | ((h & 2u) ? 0x0000FF00u : 0u) | ((h & 4u) ? 0x00FF0000u : 0u) | ((h & 8u) ? 0xFF000000u : 0u);
  return as_type<char4>(qtip_race_negate_bytes(x, mask));
}

// V2 packed pair negation: bytes of a ushort (component 0 = low byte, 1 = high byte)
static inline ushort qtip_race_negate_pair(ushort x, ushort byte_mask) {
  const ushort t = x ^ byte_mask;
  const ushort inc = ((t & (0x7F7Fu & byte_mask)) + (0x0101u & byte_mask)) ^ (t & (0x8080u & byte_mask));
  return (x & ~byte_mask) | (inc & byte_mask);
}
static inline char2 qtip_race_v2_two_sign(char2 v, ushort state) {
  ushort x = as_type<ushort>(v);
  x = (state & 0x8000u) ? qtip_race_negate_pair(x, 0xFFFFu) : x;
  x = (state & 0x4000u) ? qtip_race_negate_pair(x, 0x00FFu) : x;
  return as_type<char2>(x);
}

// Restarted V4 (17 bytes per 64 columns = 16 groups): branch-free.
//   group_in_block 0 : seq[0] | seq[1] << 8
//   group_in_block 1 : seq[0] << 8 | seq[2]
//   otherwise        : seq[g] << 8 | seq[g + 1]
static inline ushort qtip_race_v4_state(
    device const uchar* sequence,
    uint group_in_block) {
  const uint s0 = uint(sequence[0]);
  const uint a = uint(sequence[group_in_block]);
  const uint b = uint(sequence[group_in_block + 1u]);
  const uint hi = group_in_block == 1u ? s0 : (group_in_block == 0u ? b : a);
  const uint lo = group_in_block == 0u ? a : b;
  return ushort((hi << 8u) | lo);
}

// ---------------------------------------------------------------------------
// Gather of one lane's 8 left-fragment bytes (two rows x 4 columns) for a chunk
// ---------------------------------------------------------------------------

template <uint VECTOR_WIDTH, uint TRANSITION_BITS, uint DIAG>
struct QtipRaceLaneGather {
  // DIAG: 0 = exact, 1 = MXU only (no table reads), 2 = footprint 14-bit mask,
  //       3 = footprint 15-bit mask
  device const uchar* codes0;
  device const uchar* codes1;
  device const int8_t* codebook;
  bool valid0;
  bool valid1;
  uint lane_col;  // 0, 4, 8, 12

  template <typename Vector>
  METAL_FUNC void gather(uint chunk, thread Vector& fragment_values0, thread Vector& fragment_values1) const thread {
    // fragment_values0: fragment column 0 (columns lane_col + 0..3)
    // fragment_values1: fragment column 1 (columns lane_col + 16..19)
    if constexpr (DIAG == 1u || DIAG == 6u) {
      const int8_t seed = int8_t(int(chunk & 7u) - 3);
      METAL_PRAGMA_UNROLL
      for (ushort i = 0; i < 8; ++i) {
        fragment_values0[i] = int8_t(seed + int8_t(i));
        fragment_values1[i] = int8_t(seed - int8_t(i));
      }
      return;
    }
    if constexpr (VECTOR_WIDTH == 4u && TRANSITION_BITS == 12u) {
      // V4 k=3: 25-byte restart-64 blocks = [20-bit seed][15 x 12-bit symbols], MSB-first; the state after
      // group g is the 20-bit window at bit 12g, masked to L bits; base row = low 15 bits, symmetry bits above
      device const char4* codebook_vectors = reinterpret_cast<device const char4*>(codebook);
      constexpr uint L_MASK = DIAG == 13u ? 0x1FFFFu : 0xFFFFFu;
      auto state_at = [&](device const uchar* row, uint group) {
        const uint block = group >> 4u;
        const uint bit = (group & 15u) * 12u;
        device const uchar* seq = row + block * 25u + (bit >> 3u);
        const uint shift = bit & 7u;
        const uint window = (uint(seq[0]) << 24u) | (uint(seq[1]) << 16u) | (uint(seq[2]) << 8u) | uint(seq[3]);
        return (window >> (12u - shift)) & L_MASK;
      };
      auto emit = [&](uint state) {
        char4 v = codebook_vectors[state & 0x7FFFu];
        const uint h = state >> 15u;
        if constexpr (DIAG == 13u) {
          v = (h & 1u) ? -v : v;
          v = char4((h & 2u) ? -v.x : v.x, (h & 2u) ? -v.y : v.y, v.z, v.w);
        } else {
          v = char4((h & 1u) ? -v.x : v.x, (h & 2u) ? -v.y : v.y, (h & 4u) ? -v.z : v.z, (h & 8u) ? -v.w : v.w);
          v = (h & 16u) ? char4(v.y, v.x, v.z, v.w) : v;
        }
        return v;
      };
      const uint g0 = chunk * 8u + (lane_col >> 2u);
      char4 v00 = emit(state_at(codes0, g0));
      char4 v10 = emit(state_at(codes1, g0));
      char4 v01 = emit(state_at(codes0, g0 + 4u));
      char4 v11 = emit(state_at(codes1, g0 + 4u));
      v00 = valid0 ? v00 : char4(0); v01 = valid0 ? v01 : char4(0);
      v10 = valid1 ? v10 : char4(0); v11 = valid1 ? v11 : char4(0);
      fragment_values0[0] = v00.x; fragment_values0[1] = v00.y; fragment_values0[2] = v00.z; fragment_values0[3] = v00.w;
      fragment_values0[4] = v10.x; fragment_values0[5] = v10.y; fragment_values0[6] = v10.z; fragment_values0[7] = v10.w;
      fragment_values1[0] = v01.x; fragment_values1[1] = v01.y; fragment_values1[2] = v01.z; fragment_values1[3] = v01.w;
      fragment_values1[4] = v11.x; fragment_values1[5] = v11.y; fragment_values1[6] = v11.z; fragment_values1[7] = v11.w;
    } else if constexpr (VECTOR_WIDTH == 4u) {
      device const char4* codebook_vectors = reinterpret_cast<device const char4*>(codebook);
      const uint block = chunk >> 1u;
      const uint parity_base = (chunk & 1u) * 8u;
      device const uchar* seq0 = codes0 + block * 17u;
      device const uchar* seq1 = codes1 + block * 17u;
      const uint gib = parity_base + (lane_col >> 2u);
      // one seed-byte load per row per chunk instead of one per state (gib >= 2 never needs it)
      const uint seed0 = uint(seq0[0]);
      const uint seed1 = uint(seq1[0]);
      auto state_of = [&](device const uchar* seq, uint seed, uint g) {
        const uint a = uint(seq[g]);
        const uint b = uint(seq[g + 1u]);
        const uint hi = g == 1u ? seed : (g == 0u ? b : a);
        const uint lo = g == 0u ? a : b;
        return ushort((hi << 8u) | lo);
      };
      ushort s00 = state_of(seq0, seed0, gib);
      ushort s10 = state_of(seq1, seed1, gib);
      ushort s01 = state_of(seq0, seed0, gib + 4u);
      ushort s11 = state_of(seq1, seed1, gib + 4u);
      const ushort h00 = s00 >> 12u;
      const ushort h10 = s10 >> 12u;
      const ushort h01 = s01 >> 12u;
      const ushort h11 = s11 >> 12u;
      if constexpr (DIAG == 2u) {
        s00 &= 0x3FFFu; s10 &= 0x3FFFu; s01 &= 0x3FFFu; s11 &= 0x3FFFu;
      } else if constexpr (DIAG == 3u || DIAG == 8u) {
        s00 &= 0x7FFFu; s10 &= 0x7FFFu; s01 &= 0x7FFFu; s11 &= 0x7FFFu;
      } else if constexpr (DIAG == 9u) {
        s00 &= 0x0FFFu; s10 &= 0x0FFFu; s01 &= 0x0FFFu; s11 &= 0x0FFFu;
      } else if constexpr (DIAG == 11u) {
        s00 &= 0x3FFFu; s10 &= 0x3FFFu; s01 &= 0x3FFFu; s11 &= 0x3FFFu;
      }
      char4 v00 = char4(0);
      char4 v10 = char4(0);
      char4 v01 = char4(0);
      char4 v11 = char4(0);
      if constexpr (DIAG == 8u) {
        // antipodal L=16: the top state bit negates the L=15 row (packed SWAR negation)
        v00 = codebook_vectors[s00]; v10 = codebook_vectors[s10]; v01 = codebook_vectors[s01]; v11 = codebook_vectors[s11];
        v00 = qtip_race_flip_all(v00, (h00 & 8u) != 0u); v10 = qtip_race_flip_all(v10, (h10 & 8u) != 0u);
        v01 = qtip_race_flip_all(v01, (h01 & 8u) != 0u); v11 = qtip_race_flip_all(v11, (h11 & 8u) != 0u);
        v00 = valid0 ? v00 : char4(0); v01 = valid0 ? v01 : char4(0);
        v10 = valid1 ? v10 : char4(0); v11 = valid1 ? v11 : char4(0);
      } else if constexpr (DIAG == 11u) {
        // 64 KiB base table: bit 15 negates the row, bit 14 negates components 0,1
        v00 = codebook_vectors[s00]; v10 = codebook_vectors[s10]; v01 = codebook_vectors[s01]; v11 = codebook_vectors[s11];
        auto flip2 = [](char4 v, ushort h) {
          v = qtip_race_flip_all(v, (h & 8u) != 0u);
          return qtip_race_flip_lo2(v, (h & 4u) != 0u);
        };
        v00 = flip2(v00, h00); v10 = flip2(v10, h10); v01 = flip2(v01, h01); v11 = flip2(v11, h11);
        v00 = valid0 ? v00 : char4(0); v01 = valid0 ? v01 : char4(0);
        v10 = valid1 ? v10 : char4(0); v11 = valid1 ? v11 : char4(0);
      } else if constexpr (DIAG == 9u) {
        // 16 KiB base table, state bits 12..15 flip components 0..3
        v00 = codebook_vectors[s00]; v10 = codebook_vectors[s10]; v01 = codebook_vectors[s01]; v11 = codebook_vectors[s11];
        v00 = qtip_race_flip_bits(v00, h00); v10 = qtip_race_flip_bits(v10, h10);
        v01 = qtip_race_flip_bits(v01, h01); v11 = qtip_race_flip_bits(v11, h11);
        v00 = valid0 ? v00 : char4(0); v01 = valid0 ? v01 : char4(0);
        v10 = valid1 ? v10 : char4(0); v11 = valid1 ? v11 : char4(0);
      } else if constexpr (DIAG == 4u || DIAG == 5u) {
        // half-table pass: only states in this half are gathered (predicated loads)
        constexpr ushort half_bit = 0x8000u;
        constexpr ushort wanted = DIAG == 4u ? 0u : half_bit;
        if (valid0 && (s00 & half_bit) == wanted) v00 = codebook_vectors[s00];
        if (valid1 && (s10 & half_bit) == wanted) v10 = codebook_vectors[s10];
        if (valid0 && (s01 & half_bit) == wanted) v01 = codebook_vectors[s01];
        if (valid1 && (s11 & half_bit) == wanted) v11 = codebook_vectors[s11];
      } else {
        v00 = codebook_vectors[s00];
        v10 = codebook_vectors[s10];
        v01 = codebook_vectors[s01];
        v11 = codebook_vectors[s11];
        v00 = valid0 ? v00 : char4(0);
        v01 = valid0 ? v01 : char4(0);
        v10 = valid1 ? v10 : char4(0);
        v11 = valid1 ? v11 : char4(0);
      }
      fragment_values0[0] = v00.x; fragment_values0[1] = v00.y; fragment_values0[2] = v00.z; fragment_values0[3] = v00.w;
      fragment_values0[4] = v10.x; fragment_values0[5] = v10.y; fragment_values0[6] = v10.z; fragment_values0[7] = v10.w;
      fragment_values1[0] = v01.x; fragment_values1[1] = v01.y; fragment_values1[2] = v01.z; fragment_values1[3] = v01.w;
      fragment_values1[4] = v11.x; fragment_values1[5] = v11.y; fragment_values1[6] = v11.z; fragment_values1[7] = v11.w;
    } else {
      device const char2* codebook_pairs = reinterpret_cast<device const char2*>(codebook);
      const uint column = chunk * 32u + lane_col;
      ushort2 s0a = qtip_race_state_pair_v2<TRANSITION_BITS>(codes0, column);
      ushort2 s1a = qtip_race_state_pair_v2<TRANSITION_BITS>(codes1, column);
      ushort2 s0b = qtip_race_state_pair_v2<TRANSITION_BITS>(codes0, column + 16u);
      ushort2 s1b = qtip_race_state_pair_v2<TRANSITION_BITS>(codes1, column + 16u);
      const ushort2 f0a = s0a, f1a = s1a, f0b = s0b, f1b = s1b;   // full states for the sign bits
      if constexpr (DIAG == 2u) {
        s0a &= 0x3FFFu; s1a &= 0x3FFFu; s0b &= 0x3FFFu; s1b &= 0x3FFFu;
      } else if constexpr (DIAG == 3u) {
        s0a &= 0x7FFFu; s1a &= 0x7FFFu; s0b &= 0x7FFFu; s1b &= 0x7FFFu;
      } else if constexpr (DIAG == 14u) {
        s0a &= 0x3FFFu; s1a &= 0x3FFFu; s0b &= 0x3FFFu; s1b &= 0x3FFFu;
      }
      char2 p0a0 = codebook_pairs[s0a.x];
      char2 p0a1 = codebook_pairs[s0a.y];
      char2 p1a0 = codebook_pairs[s1a.x];
      char2 p1a1 = codebook_pairs[s1a.y];
      char2 p0b0 = codebook_pairs[s0b.x];
      char2 p0b1 = codebook_pairs[s0b.y];
      char2 p1b0 = codebook_pairs[s1b.x];
      char2 p1b1 = codebook_pairs[s1b.y];
      if constexpr (DIAG == 14u) {
        p0a0 = qtip_race_v2_two_sign(p0a0, f0a.x); p0a1 = qtip_race_v2_two_sign(p0a1, f0a.y);
        p1a0 = qtip_race_v2_two_sign(p1a0, f1a.x); p1a1 = qtip_race_v2_two_sign(p1a1, f1a.y);
        p0b0 = qtip_race_v2_two_sign(p0b0, f0b.x); p0b1 = qtip_race_v2_two_sign(p0b1, f0b.y);
        p1b0 = qtip_race_v2_two_sign(p1b0, f1b.x); p1b1 = qtip_race_v2_two_sign(p1b1, f1b.y);
      }
      p0a0 = valid0 ? p0a0 : char2(0); p0a1 = valid0 ? p0a1 : char2(0);
      p0b0 = valid0 ? p0b0 : char2(0); p0b1 = valid0 ? p0b1 : char2(0);
      p1a0 = valid1 ? p1a0 : char2(0); p1a1 = valid1 ? p1a1 : char2(0);
      p1b0 = valid1 ? p1b0 : char2(0); p1b1 = valid1 ? p1b1 : char2(0);
      fragment_values0[0] = p0a0.x; fragment_values0[1] = p0a0.y; fragment_values0[2] = p0a1.x; fragment_values0[3] = p0a1.y;
      fragment_values0[4] = p1a0.x; fragment_values0[5] = p1a0.y; fragment_values0[6] = p1a1.x; fragment_values0[7] = p1a1.y;
      fragment_values1[0] = p0b0.x; fragment_values1[1] = p0b0.y; fragment_values1[2] = p0b1.x; fragment_values1[3] = p0b1.y;
      fragment_values1[4] = p1b0.x; fragment_values1[5] = p1b0.y; fragment_values1[6] = p1b1.x; fragment_values1[7] = p1b1.y;
    }
  }
};

// ---------------------------------------------------------------------------
// B32 / B64: one row fragment (16 rows) per SIMDgroup, device-tensor activations
// ---------------------------------------------------------------------------

template <
    uint COLS,
    uint VECTOR_WIDTH,
    uint TRANSITION_BITS,
    uint ROW_SIMDGROUPS,
    uint PREFETCH,
    uint DIAG,
    bool GATHER_ONLY,
    uint ROW_FRAGMENTS = 1>
static inline void qtip_race_dt(
    device const uchar* codes,
    device const int8_t* codebook,
    device const int8_t* activations,
    device const float* activation_scales,
    device const half* scales,
    device const ushort* gains_bf16,
    device bfloat* output,
    device int32_t* partials,
    float codebook_scale,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint active_batch,
    uint row_tile,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::OperandFragment<int8_t, ROW_FRAGMENTS, 2, Ops>;
  using Accumulator = uzu::matmul::Fragment<int32_t, ROW_FRAGMENTS, COLS / 16, Ops>;
  using Format = uzu::matmul::IntegerFormat<8, uzu::matmul::Signedness::Signed>;
  using Right = uzu::matmul::DeviceTensorOperand<Format>;

  const uint row_base = row_tile * (ROW_SIMDGROUPS * ROW_FRAGMENTS * 16u) +
      thread_context.simdgroup_index * ROW_FRAGMENTS * 16u;
  const short2 lane_position = Ops::get_position(thread_context.simd_lane_id);
  const uint columns = groups * VECTOR_WIDTH;
  const uint chunk_count = columns / 32u;

  QtipRaceLaneGather<VECTOR_WIDTH, TRANSITION_BITS, DIAG> lanes[ROW_FRAGMENTS];
  METAL_PRAGMA_UNROLL
  for (ushort fragment_row = 0; fragment_row < ROW_FRAGMENTS; ++fragment_row) {
    const uint row0 = row_base + uint(lane_position.y) + uint(fragment_row) * 16u;
    const uint row1 = row0 + 8u;
    lanes[fragment_row].codes0 = codes + min(row0, rows - 1u) * bytes_per_row;
    lanes[fragment_row].codes1 = codes + min(row1, rows - 1u) * bytes_per_row;
    lanes[fragment_row].codebook = codebook;
    lanes[fragment_row].valid0 = row0 < rows;
    lanes[fragment_row].valid1 = row1 < rows;
    lanes[fragment_row].lane_col = uint(lane_position.x);
  }
  auto gather_tile = [&](uint chunk, thread LeftTile& tile) {
    METAL_PRAGMA_UNROLL
    for (ushort fragment_row = 0; fragment_row < ROW_FRAGMENTS; ++fragment_row) {
      lanes[fragment_row].gather(chunk, tile.fragment_at(fragment_row, 0), tile.fragment_at(fragment_row, 1));
    }
  };

  Accumulator accumulator;
  const int row_stride_bytes = int(columns);

  if constexpr (GATHER_ONLY) {
    // gather + decode only; fold bytes so the loads cannot be eliminated
    int fold = 0;
    for (uint chunk = 0; chunk < chunk_count; ++chunk) {
      LeftTile tile;
      gather_tile(chunk, tile);
      thread int8_t* bytes = tile.elements();
      METAL_PRAGMA_UNROLL
      for (ushort i = 0; i < 16 * ROW_FRAGMENTS; ++i) {
        fold += int(bytes[i]);
      }
    }
    accumulator.clear();
    accumulator.map_coords(thread_context.simd_lane_id, [&](short, short, int32_t) { return fold; });
  } else if constexpr (PREFETCH == 0u) {
    for (uint chunk = 0; chunk < chunk_count; ++chunk) {
      LeftTile tile;
      gather_tile(chunk, tile);
      const Right right{activations + ((DIAG == 6u || DIAG == 7u) ? 0u : chunk * 32u), row_stride_bytes};
      if (chunk == 0u) {
        uzu::matmul::fragment_mm(accumulator, tile, right);
      } else {
        uzu::matmul::fragment_mma(accumulator, tile, right);
      }
    }
  } else if constexpr (PREFETCH == 1u) {
    LeftTile current;
    gather_tile(0u, current);
    for (uint chunk = 0; chunk < chunk_count; ++chunk) {
      LeftTile next;
      if (chunk + 1u < chunk_count) {
        gather_tile(chunk + 1u, next);
      }
      const Right right{activations + ((DIAG == 6u || DIAG == 7u) ? 0u : chunk * 32u), row_stride_bytes};
      if (chunk == 0u) {
        uzu::matmul::fragment_mm(accumulator, current, right);
      } else {
        uzu::matmul::fragment_mma(accumulator, current, right);
      }
      current = next;
    }
  } else if constexpr (PREFETCH == 2u) {
    LeftTile tile0;
    LeftTile tile1;
    gather_tile(0u, tile0);
    if (chunk_count > 1u) {
      gather_tile(1u, tile1);
    }
    for (uint chunk = 0; chunk < chunk_count; chunk += 2u) {
      LeftTile next0;
      LeftTile next1;
      if (chunk + 2u < chunk_count) {
        gather_tile(chunk + 2u, next0);
      }
      if (chunk + 3u < chunk_count) {
        gather_tile(chunk + 3u, next1);
      }
      {
        const Right right{activations + ((DIAG == 6u || DIAG == 7u) ? 0u : chunk * 32u), row_stride_bytes};
        if (chunk == 0u) {
          uzu::matmul::fragment_mm(accumulator, tile0, right);
        } else {
          uzu::matmul::fragment_mma(accumulator, tile0, right);
        }
      }
      if (chunk + 1u < chunk_count) {
        const Right right{activations + ((DIAG == 6u || DIAG == 7u) ? 0u : (chunk + 1u) * 32u), row_stride_bytes};
        uzu::matmul::fragment_mma(accumulator, tile1, right);
      }
      tile0 = next0;
      tile1 = next1;
    }
  } else {
    // PREFETCH == 4: four chunks in flight
    LeftTile tile0;
    LeftTile tile1;
    LeftTile tile2;
    LeftTile tile3;
    gather_tile(0u, tile0);
    if (chunk_count > 1u) {
      gather_tile(1u, tile1);
    }
    if (chunk_count > 2u) {
      gather_tile(2u, tile2);
    }
    if (chunk_count > 3u) {
      gather_tile(3u, tile3);
    }
    for (uint chunk = 0; chunk < chunk_count; chunk += 4u) {
      LeftTile next0;
      LeftTile next1;
      LeftTile next2;
      LeftTile next3;
      if (chunk + 4u < chunk_count) {
        gather_tile(chunk + 4u, next0);
      }
      {
        const Right right{activations + ((DIAG == 6u || DIAG == 7u) ? 0u : chunk * 32u), row_stride_bytes};
        if (chunk == 0u) {
          uzu::matmul::fragment_mm(accumulator, tile0, right);
        } else {
          uzu::matmul::fragment_mma(accumulator, tile0, right);
        }
      }
      if (chunk + 5u < chunk_count) {
        gather_tile(chunk + 5u, next1);
      }
      if (chunk + 1u < chunk_count) {
        const Right right{activations + ((DIAG == 6u || DIAG == 7u) ? 0u : (chunk + 1u) * 32u), row_stride_bytes};
        uzu::matmul::fragment_mma(accumulator, tile1, right);
      }
      if (chunk + 6u < chunk_count) {
        gather_tile(chunk + 6u, next2);
      }
      if (chunk + 2u < chunk_count) {
        const Right right{activations + ((DIAG == 6u || DIAG == 7u) ? 0u : (chunk + 2u) * 32u), row_stride_bytes};
        uzu::matmul::fragment_mma(accumulator, tile2, right);
      }
      if (chunk + 7u < chunk_count) {
        gather_tile(chunk + 7u, next3);
      }
      if (chunk + 3u < chunk_count) {
        const Right right{activations + ((DIAG == 6u || DIAG == 7u) ? 0u : (chunk + 3u) * 32u), row_stride_bytes};
        uzu::matmul::fragment_mma(accumulator, tile3, right);
      }
      tile0 = next0;
      tile1 = next1;
      tile2 = next2;
      tile3 = next3;
    }
  }

  accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short col, int32_t value) {
    const uint absolute_row = row_base + uint(row);
    if (absolute_row < rows && uint(col) < active_batch) {
      const uint index = uint(col) * rows + absolute_row;
      if constexpr (DIAG == 4u) {
        partials[index] = value;
      } else {
        int32_t total = value;
        if constexpr (DIAG == 5u) {
          total += partials[index];
        }
        const float gain = as_type<float>(uint(gains_bf16[absolute_row]) << 16u);
        const float weight_scale = float(scales[absolute_row]) * gain * codebook_scale;
        output[index] = bfloat(float(total) * weight_scale * activation_scales[uint(col)]);
      }
    }
    return value;
  });
}

// ---------------------------------------------------------------------------
// B16T: tokens as the MXU M dimension (16), weight rows as the paired N operand.
// out^T[token][row] = A[token][k] . W[row][k]. The gathered weight tile is the
// right operand read transposed (legal N pairing), so 16-token execution does
// no padded MXU work and needs no transpose pass. ROW_FRAGMENTS (2 or 4) sets
// rows per SIMDgroup (32 or 64), i.e. gathers in flight per chunk.
// ---------------------------------------------------------------------------

template <uint VECTOR_WIDTH, uint TRANSITION_BITS, uint ROW_FRAGMENTS, uint ROW_SIMDGROUPS, uint PREFETCH, uint HALF = 0>
static inline void qtip_race_b16t(
    device const uchar* codes,
    device const int8_t* codebook,
    device const int8_t* activations,
    device const float* activation_scales,
    device const half* scales,
    device const ushort* gains_bf16,
    device bfloat* output,
    device int32_t* partials,
    float codebook_scale,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint active_batch,
    uint row_tile,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::OperandFragment<int8_t, 1, 2, Ops>;                              // 16 tokens x 32 k
  using RightTile = uzu::matmul::OperandFragment<int8_t, 2, ROW_FRAGMENTS, Ops, uzu::matmul::ReadTranspose>;  // stored [rows x k]
  using Accumulator = uzu::matmul::Fragment<int32_t, 1, ROW_FRAGMENTS, Ops>;                     // 16 tokens x rows

  const uint row_base = row_tile * (ROW_SIMDGROUPS * ROW_FRAGMENTS * 16u) +
      thread_context.simdgroup_index * ROW_FRAGMENTS * 16u;
  const short2 lane_position = Ops::get_position(thread_context.simd_lane_id);
  const uint columns = groups * VECTOR_WIDTH;
  const uint chunk_count = columns / 32u;

  QtipRaceLaneGather<VECTOR_WIDTH, TRANSITION_BITS, HALF> lanes[ROW_FRAGMENTS];
  METAL_PRAGMA_UNROLL
  for (ushort fragment_row = 0; fragment_row < ROW_FRAGMENTS; ++fragment_row) {
    const uint row0 = row_base + uint(lane_position.y) + uint(fragment_row) * 16u;
    const uint row1 = row0 + 8u;
    lanes[fragment_row].codes0 = codes + min(row0, rows - 1u) * bytes_per_row;
    lanes[fragment_row].codes1 = codes + min(row1, rows - 1u) * bytes_per_row;
    lanes[fragment_row].codebook = codebook;
    lanes[fragment_row].valid0 = row0 < rows;
    lanes[fragment_row].valid1 = row1 < rows;
    lanes[fragment_row].lane_col = uint(lane_position.x);
  }
  auto gather_tile = [&](uint chunk, thread RightTile& tile) {
    METAL_PRAGMA_UNROLL
    for (ushort fragment_row = 0; fragment_row < ROW_FRAGMENTS; ++fragment_row) {
      lanes[fragment_row].gather(chunk, tile.fragment_at(fragment_row, 0), tile.fragment_at(fragment_row, 1));
    }
  };
  auto load_left = [&](uint chunk, thread LeftTile& left) {
    left.load_from(
        thread_context.simd_lane_id,
        uzu::matmul::fragment_source(activations + chunk * 32u, int(columns)));
  };

  Accumulator accumulator;
  if constexpr (PREFETCH == 0u) {
    for (uint chunk = 0; chunk < chunk_count; ++chunk) {
      RightTile tile;
      gather_tile(chunk, tile);
      LeftTile left;
      load_left(chunk, left);
      if (chunk == 0u) {
        uzu::matmul::fragment_mm(accumulator, left, tile);
      } else {
        uzu::matmul::fragment_mma(accumulator, left, tile);
      }
    }
  } else if constexpr (PREFETCH == 1u) {
    RightTile current;
    gather_tile(0u, current);
    for (uint chunk = 0; chunk < chunk_count; ++chunk) {
      RightTile next;
      if (chunk + 1u < chunk_count) {
        gather_tile(chunk + 1u, next);
      }
      LeftTile left;
      load_left(chunk, left);
      if (chunk == 0u) {
        uzu::matmul::fragment_mm(accumulator, left, current);
      } else {
        uzu::matmul::fragment_mma(accumulator, left, current);
      }
      current = next;
    }
  } else {
    RightTile tile0;
    RightTile tile1;
    gather_tile(0u, tile0);
    if (chunk_count > 1u) {
      gather_tile(1u, tile1);
    }
    for (uint chunk = 0; chunk < chunk_count; chunk += 2u) {
      RightTile next0;
      RightTile next1;
      if (chunk + 2u < chunk_count) {
        gather_tile(chunk + 2u, next0);
      }
      if (chunk + 3u < chunk_count) {
        gather_tile(chunk + 3u, next1);
      }
      {
        LeftTile left;
        load_left(chunk, left);
        if (chunk == 0u) {
          uzu::matmul::fragment_mm(accumulator, left, tile0);
        } else {
          uzu::matmul::fragment_mma(accumulator, left, tile0);
        }
      }
      if (chunk + 1u < chunk_count) {
        LeftTile left;
        load_left(chunk + 1u, left);
        uzu::matmul::fragment_mma(accumulator, left, tile1);
      }
      tile0 = next0;
      tile1 = next1;
    }
  }

  accumulator.map_coords(thread_context.simd_lane_id, [&](short token, short row, int32_t value) {
    const uint absolute_row = row_base + uint(row);
    if (absolute_row < rows && uint(token) < active_batch) {
      const uint index = uint(token) * rows + absolute_row;
      if constexpr (HALF == 4u) {
        partials[index] = value;
      } else {
        int32_t total = value;
        if constexpr (HALF == 5u) {
          total += partials[index];
        }
        const float gain = as_type<float>(uint(gains_bf16[absolute_row]) << 16u);
        const float weight_scale = float(scales[absolute_row]) * gain * codebook_scale;
        output[index] = bfloat(float(total) * weight_scale * activation_scales[uint(token)]);
      }
    }
    return value;
  });
}

// ---------------------------------------------------------------------------
// V4 component split ("CS"): the Q8 V4 table is stored as two 128 KiB char2
// tables (components 0,1 | components 2,3) and the activations as two K-packed
// halves (columns 4g,4g+1 | 4g+2,4g+3). Pass 0 multiplies the (0,1) half,
// pass 1 the (2,3) half; both passes are full-density MXU work over K/2, and
// each pass only touches a 128 KiB table. Pass 0 writes int32 partials, pass 1
// adds them and applies the epilogue. Two dispatches keep every SIMDgroup on
// the GPU in the same table half at any time.
// ---------------------------------------------------------------------------

template <uint COLS, uint ROW_SIMDGROUPS, uint ROW_FRAGMENTS, uint PREFETCH, uint PASS>
static inline void qtip_race_v4_cs(
    device const uchar* codes,
    device const int8_t* codebook_split,
    device const int8_t* activations_half,
    device const float* activation_scales,
    device const half* scales,
    device const ushort* gains_bf16,
    device bfloat* output,
    device int32_t* partials,
    float codebook_scale,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint active_batch,
    uint row_tile,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::OperandFragment<int8_t, ROW_FRAGMENTS, 2, Ops>;
  using Accumulator = uzu::matmul::Fragment<int32_t, ROW_FRAGMENTS, COLS / 16, Ops>;
  using Format = uzu::matmul::IntegerFormat<8, uzu::matmul::Signedness::Signed>;
  using Right = uzu::matmul::DeviceTensorOperand<Format>;

  const uint row_base = row_tile * (ROW_SIMDGROUPS * ROW_FRAGMENTS * 16u) +
      thread_context.simdgroup_index * ROW_FRAGMENTS * 16u;
  const short2 lane_position = Ops::get_position(thread_context.simd_lane_id);
  const uint half_columns = groups * 2u;       // K'
  const uint chunk_count = half_columns / 32u;  // groups / 16
  device const char2* table = reinterpret_cast<device const char2*>(codebook_split) + (PASS == 0u ? 0u : 65536u);

  device const uchar* row_codes[ROW_FRAGMENTS * 2];
  bool valid[ROW_FRAGMENTS * 2];
  METAL_PRAGMA_UNROLL
  for (ushort fragment_row = 0; fragment_row < ROW_FRAGMENTS; ++fragment_row) {
    METAL_PRAGMA_UNROLL
    for (ushort local_row = 0; local_row < 2; ++local_row) {
      const uint row = row_base + uint(lane_position.y) + uint(fragment_row) * 16u + uint(local_row) * 8u;
      valid[fragment_row * 2 + local_row] = row < rows;
      row_codes[fragment_row * 2 + local_row] = codes + min(row, rows - 1u) * bytes_per_row;
    }
  }
  const uint lane_col = uint(lane_position.x);  // 0, 4, 8, 12 within the K' chunk

  auto gather_tile = [&](uint chunk, thread LeftTile& tile) {
    METAL_PRAGMA_UNROLL
    for (ushort fragment_col = 0; fragment_col < 2; ++fragment_col) {
      // K' columns k'..k'+3 = groups g0 (2 values) and g0 + 1 (2 values); g0 is even so both lie in one block
      const uint g0 = (chunk * 32u + lane_col + uint(fragment_col) * 16u) >> 1u;
      const uint block = g0 >> 4u;
      const uint gib = g0 & 15u;
      METAL_PRAGMA_UNROLL
      for (ushort fragment_row = 0; fragment_row < ROW_FRAGMENTS; ++fragment_row) {
        thread auto& fragment_values = tile.fragment_at(fragment_row, fragment_col);
        METAL_PRAGMA_UNROLL
        for (ushort local_row = 0; local_row < 2; ++local_row) {
          const ushort i = fragment_row * 2 + local_row;
          device const uchar* seq = row_codes[i] + block * 17u;
          const ushort s0 = qtip_race_v4_state(seq, gib);
          const ushort s1 = qtip_race_v4_state(seq, gib + 1u);
          char2 v0 = table[s0];
          char2 v1 = table[s1];
          v0 = valid[i] ? v0 : char2(0);
          v1 = valid[i] ? v1 : char2(0);
          fragment_values[local_row * 4 + 0] = v0.x;
          fragment_values[local_row * 4 + 1] = v0.y;
          fragment_values[local_row * 4 + 2] = v1.x;
          fragment_values[local_row * 4 + 3] = v1.y;
        }
      }
    }
  };

  Accumulator accumulator;
  const int row_stride_bytes = int(half_columns);
  if constexpr (PREFETCH == 0u) {
    for (uint chunk = 0; chunk < chunk_count; ++chunk) {
      LeftTile tile;
      gather_tile(chunk, tile);
      const Right right{activations_half + chunk * 32u, row_stride_bytes};
      if (chunk == 0u) {
        uzu::matmul::fragment_mm(accumulator, tile, right);
      } else {
        uzu::matmul::fragment_mma(accumulator, tile, right);
      }
    }
  } else {
    LeftTile tile0;
    LeftTile tile1;
    gather_tile(0u, tile0);
    if (chunk_count > 1u) {
      gather_tile(1u, tile1);
    }
    for (uint chunk = 0; chunk < chunk_count; chunk += 2u) {
      LeftTile next0;
      LeftTile next1;
      if (chunk + 2u < chunk_count) {
        gather_tile(chunk + 2u, next0);
      }
      if (chunk + 3u < chunk_count) {
        gather_tile(chunk + 3u, next1);
      }
      {
        const Right right{activations_half + chunk * 32u, row_stride_bytes};
        if (chunk == 0u) {
          uzu::matmul::fragment_mm(accumulator, tile0, right);
        } else {
          uzu::matmul::fragment_mma(accumulator, tile0, right);
        }
      }
      if (chunk + 1u < chunk_count) {
        const Right right{activations_half + (chunk + 1u) * 32u, row_stride_bytes};
        uzu::matmul::fragment_mma(accumulator, tile1, right);
      }
      tile0 = next0;
      tile1 = next1;
    }
  }

  accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short col, int32_t value) {
    const uint absolute_row = row_base + uint(row);
    if (absolute_row < rows && uint(col) < active_batch) {
      const uint index = uint(col) * rows + absolute_row;
      if constexpr (PASS == 0u) {
        partials[index] = value;
      } else {
        const int32_t total = value + partials[index];
        const float gain = as_type<float>(uint(gains_bf16[absolute_row]) << 16u);
        const float weight_scale = float(scales[absolute_row]) * gain * codebook_scale;
        output[index] = bfloat(float(total) * weight_scale * activation_scales[uint(col)]);
      }
    }
    return value;
  });
}

// transposed 16-token variant of the component split (tokens = M, rows = paired N)
template <uint ROW_FRAGMENTS, uint ROW_SIMDGROUPS, uint PREFETCH, uint PASS>
static inline void qtip_race_v4_cs_b16t(
    device const uchar* codes,
    device const int8_t* codebook_split,
    device const int8_t* activations_half,
    device const float* activation_scales,
    device const half* scales,
    device const ushort* gains_bf16,
    device bfloat* output,
    device int32_t* partials,
    float codebook_scale,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint active_batch,
    uint row_tile,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::OperandFragment<int8_t, 1, 2, Ops>;
  using RightTile = uzu::matmul::OperandFragment<int8_t, 2, ROW_FRAGMENTS, Ops, uzu::matmul::ReadTranspose>;
  using Accumulator = uzu::matmul::Fragment<int32_t, 1, ROW_FRAGMENTS, Ops>;

  const uint row_base = row_tile * (ROW_SIMDGROUPS * ROW_FRAGMENTS * 16u) +
      thread_context.simdgroup_index * ROW_FRAGMENTS * 16u;
  const short2 lane_position = Ops::get_position(thread_context.simd_lane_id);
  const uint half_columns = groups * 2u;
  const uint chunk_count = half_columns / 32u;
  device const char2* table = reinterpret_cast<device const char2*>(codebook_split) + (PASS == 0u ? 0u : 65536u);

  device const uchar* row_codes[ROW_FRAGMENTS * 2];
  bool valid[ROW_FRAGMENTS * 2];
  METAL_PRAGMA_UNROLL
  for (ushort fragment_row = 0; fragment_row < ROW_FRAGMENTS; ++fragment_row) {
    METAL_PRAGMA_UNROLL
    for (ushort local_row = 0; local_row < 2; ++local_row) {
      const uint row = row_base + uint(lane_position.y) + uint(fragment_row) * 16u + uint(local_row) * 8u;
      valid[fragment_row * 2 + local_row] = row < rows;
      row_codes[fragment_row * 2 + local_row] = codes + min(row, rows - 1u) * bytes_per_row;
    }
  }
  const uint lane_col = uint(lane_position.x);

  auto gather_tile = [&](uint chunk, thread RightTile& tile) {
    METAL_PRAGMA_UNROLL
    for (ushort fragment_col = 0; fragment_col < 2; ++fragment_col) {
      const uint g0 = (chunk * 32u + lane_col + uint(fragment_col) * 16u) >> 1u;
      const uint block = g0 >> 4u;
      const uint gib = g0 & 15u;
      METAL_PRAGMA_UNROLL
      for (ushort fragment_row = 0; fragment_row < ROW_FRAGMENTS; ++fragment_row) {
        thread auto& fragment_values = tile.fragment_at(fragment_row, fragment_col);
        METAL_PRAGMA_UNROLL
        for (ushort local_row = 0; local_row < 2; ++local_row) {
          const ushort i = fragment_row * 2 + local_row;
          device const uchar* seq = row_codes[i] + block * 17u;
          const ushort s0 = qtip_race_v4_state(seq, gib);
          const ushort s1 = qtip_race_v4_state(seq, gib + 1u);
          char2 v0 = table[s0];
          char2 v1 = table[s1];
          v0 = valid[i] ? v0 : char2(0);
          v1 = valid[i] ? v1 : char2(0);
          fragment_values[local_row * 4 + 0] = v0.x;
          fragment_values[local_row * 4 + 1] = v0.y;
          fragment_values[local_row * 4 + 2] = v1.x;
          fragment_values[local_row * 4 + 3] = v1.y;
        }
      }
    }
  };
  auto load_left = [&](uint chunk, thread LeftTile& left) {
    left.load_from(
        thread_context.simd_lane_id,
        uzu::matmul::fragment_source(activations_half + chunk * 32u, int(half_columns)));
  };

  Accumulator accumulator;
  if constexpr (PREFETCH == 0u) {
    for (uint chunk = 0; chunk < chunk_count; ++chunk) {
      RightTile tile;
      gather_tile(chunk, tile);
      LeftTile left;
      load_left(chunk, left);
      if (chunk == 0u) {
        uzu::matmul::fragment_mm(accumulator, left, tile);
      } else {
        uzu::matmul::fragment_mma(accumulator, left, tile);
      }
    }
  } else {
    RightTile current;
    gather_tile(0u, current);
    for (uint chunk = 0; chunk < chunk_count; ++chunk) {
      RightTile next;
      if (chunk + 1u < chunk_count) {
        gather_tile(chunk + 1u, next);
      }
      LeftTile left;
      load_left(chunk, left);
      if (chunk == 0u) {
        uzu::matmul::fragment_mm(accumulator, left, current);
      } else {
        uzu::matmul::fragment_mma(accumulator, left, current);
      }
      current = next;
    }
  }

  accumulator.map_coords(thread_context.simd_lane_id, [&](short token, short row, int32_t value) {
    const uint absolute_row = row_base + uint(row);
    if (absolute_row < rows && uint(token) < active_batch) {
      const uint index = uint(token) * rows + absolute_row;
      if constexpr (PASS == 0u) {
        partials[index] = value;
      } else {
        const int32_t total = value + partials[index];
        const float gain = as_type<float>(uint(gains_bf16[absolute_row]) << 16u);
        const float weight_scale = float(scales[absolute_row]) * gain * codebook_scale;
        output[index] = bfloat(float(total) * weight_scale * activation_scales[uint(token)]);
      }
    }
    return value;
  });
}

// ---------------------------------------------------------------------------
// AS: activation staging. Each threadgroup copies a [COLS tokens x 256 k] block
// of the signed-A8 activations into threadgroup memory (row stride 272 bytes,
// bank-conflict free for the fragment reads) once per 8 chunks; every SIMDgroup
// then feeds the MXU from threadgroup memory through the register operand path
// (char4 reads), so activation traffic no longer competes with the table
// gathers in the L1 / L2 path. CS_PASS: 0 = full V4 (char4 table), 1 / 2 =
// component-split pass on the (0,1) / (2,3) half-table and K-packed activations.
// ---------------------------------------------------------------------------

#define QTIP_RACE_AS_STRIDE 272u
#define QTIP_RACE_AS_BLOCK 256u

template <uint COLS, uint VECTOR_WIDTH, uint TRANSITION_BITS, uint ROW_SIMDGROUPS, uint ROW_FRAGMENTS, uint CS_PASS, uint PREFETCH, uint DIAG = 0>
static inline void qtip_race_as(
    device const uchar* codes,
    device const int8_t* codebook,
    device const int8_t* activations,
    device const float* activation_scales,
    device const half* scales,
    device const ushort* gains_bf16,
    device bfloat* output,
    device int32_t* partials,
    threadgroup int8_t* staging,  // COLS * QTIP_RACE_AS_STRIDE bytes
    float codebook_scale,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint active_batch,
    uint row_tile,
    uint thread_index,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::OperandFragment<int8_t, ROW_FRAGMENTS, 2, Ops>;
  using RightTile = uzu::matmul::OperandFragment<int8_t, 2, COLS / 16, Ops, uzu::matmul::ReadTranspose>;
  using Accumulator = uzu::matmul::Fragment<int32_t, ROW_FRAGMENTS, COLS / 16, Ops>;
  constexpr uint THREADS = ROW_SIMDGROUPS * 32u;
  constexpr bool CS = CS_PASS != 0u;

  const uint row_base = row_tile * (ROW_SIMDGROUPS * ROW_FRAGMENTS * 16u) +
      thread_context.simdgroup_index * ROW_FRAGMENTS * 16u;
  const short2 lane_position = Ops::get_position(thread_context.simd_lane_id);
  const uint k_columns = CS ? groups * 2u : groups * VECTOR_WIDTH;  // activation row length
  const uint chunk_count = k_columns / 32u;
  const uint lane_col = uint(lane_position.x);

  QtipRaceLaneGather<VECTOR_WIDTH, TRANSITION_BITS, DIAG> lanes[ROW_FRAGMENTS];
  device const uchar* row_codes[ROW_FRAGMENTS * 2];
  bool valid[ROW_FRAGMENTS * 2];
  METAL_PRAGMA_UNROLL
  for (ushort fragment_row = 0; fragment_row < ROW_FRAGMENTS; ++fragment_row) {
    const uint row0 = row_base + uint(lane_position.y) + uint(fragment_row) * 16u;
    const uint row1 = row0 + 8u;
    lanes[fragment_row].codes0 = codes + min(row0, rows - 1u) * bytes_per_row;
    lanes[fragment_row].codes1 = codes + min(row1, rows - 1u) * bytes_per_row;
    lanes[fragment_row].codebook = codebook;
    lanes[fragment_row].valid0 = row0 < rows;
    lanes[fragment_row].valid1 = row1 < rows;
    lanes[fragment_row].lane_col = lane_col;
    valid[fragment_row * 2] = row0 < rows;
    valid[fragment_row * 2 + 1] = row1 < rows;
    row_codes[fragment_row * 2] = lanes[fragment_row].codes0;
    row_codes[fragment_row * 2 + 1] = lanes[fragment_row].codes1;
  }
  device const char2* half_table = reinterpret_cast<device const char2*>(codebook) + (CS_PASS == 2u ? 65536u : 0u);

  auto gather_tile = [&](uint chunk, thread LeftTile& tile) {
    if constexpr (!CS) {
      METAL_PRAGMA_UNROLL
      for (ushort fragment_row = 0; fragment_row < ROW_FRAGMENTS; ++fragment_row) {
        lanes[fragment_row].gather(chunk, tile.fragment_at(fragment_row, 0), tile.fragment_at(fragment_row, 1));
      }
    } else {
      METAL_PRAGMA_UNROLL
      for (ushort fragment_col = 0; fragment_col < 2; ++fragment_col) {
        const uint g0 = (chunk * 32u + lane_col + uint(fragment_col) * 16u) >> 1u;
        const uint block = g0 >> 4u;
        const uint gib = g0 & 15u;
        METAL_PRAGMA_UNROLL
        for (ushort fragment_row = 0; fragment_row < ROW_FRAGMENTS; ++fragment_row) {
          thread auto& fragment_values = tile.fragment_at(fragment_row, fragment_col);
          METAL_PRAGMA_UNROLL
          for (ushort local_row = 0; local_row < 2; ++local_row) {
            const ushort i = fragment_row * 2 + local_row;
            device const uchar* seq = row_codes[i] + block * 17u;
            const ushort s0 = qtip_race_v4_state(seq, gib);
            const ushort s1 = qtip_race_v4_state(seq, gib + 1u);
            char2 v0 = half_table[s0];
            char2 v1 = half_table[s1];
            v0 = valid[i] ? v0 : char2(0);
            v1 = valid[i] ? v1 : char2(0);
            fragment_values[local_row * 4 + 0] = v0.x;
            fragment_values[local_row * 4 + 1] = v0.y;
            fragment_values[local_row * 4 + 2] = v1.x;
            fragment_values[local_row * 4 + 3] = v1.y;
          }
        }
      }
    }
  };

  // right operand (K x N, read transposed): storage [N fragments][K fragments], element (n, k)
  auto load_right = [&](uint local_chunk, thread RightTile& tile) {
    const uint k_base = local_chunk * 32u;
    METAL_PRAGMA_UNROLL
    for (ushort n_fragment = 0; n_fragment < COLS / 16; ++n_fragment) {
      METAL_PRAGMA_UNROLL
      for (ushort k_fragment = 0; k_fragment < 2; ++k_fragment) {
        thread auto& fragment_values = tile.fragment_at(n_fragment, k_fragment);
        METAL_PRAGMA_UNROLL
        for (ushort local_row = 0; local_row < 2; ++local_row) {
          const uint token = uint(lane_position.y) + uint(local_row) * 8u + uint(n_fragment) * 16u;
          const uint k = k_base + lane_col + uint(k_fragment) * 16u;
          const char4 values = *reinterpret_cast<threadgroup const char4*>(staging + token * QTIP_RACE_AS_STRIDE + k);
          fragment_values[local_row * 4 + 0] = values.x;
          fragment_values[local_row * 4 + 1] = values.y;
          fragment_values[local_row * 4 + 2] = values.z;
          fragment_values[local_row * 4 + 3] = values.w;
        }
      }
    }
  };

  Accumulator accumulator;
  LeftTile next_tile;
  if constexpr (PREFETCH != 0u) {
    gather_tile(0u, next_tile);
  }
  for (uint block_base = 0; block_base < chunk_count; block_base += QTIP_RACE_AS_BLOCK / 32u) {
    // stage [COLS x 256] activations: 16 bytes per thread per step
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint block_k = block_base * 32u;
    for (uint index = thread_index; index < COLS * (QTIP_RACE_AS_BLOCK / 16u); index += THREADS) {
      const uint token = index / (QTIP_RACE_AS_BLOCK / 16u);
      const uint k16 = (index - token * (QTIP_RACE_AS_BLOCK / 16u)) * 16u;
      const uint4 values = *reinterpret_cast<device const uint4*>(activations + token * k_columns + block_k + k16);
      *reinterpret_cast<threadgroup uint4*>(staging + token * QTIP_RACE_AS_STRIDE + k16) = values;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint block_chunks = min(chunk_count - block_base, QTIP_RACE_AS_BLOCK / 32u);
    for (uint local_chunk = 0; local_chunk < block_chunks; ++local_chunk) {
      const uint chunk = block_base + local_chunk;
      LeftTile tile;
      if constexpr (PREFETCH != 0u) {
        tile = next_tile;
        if (chunk + 1u < chunk_count) {
          gather_tile(chunk + 1u, next_tile);
        }
      } else {
        gather_tile(chunk, tile);
      }
      RightTile right;
      load_right(local_chunk, right);
      if (chunk == 0u) {
        uzu::matmul::fragment_mm(accumulator, tile, right);
      } else {
        uzu::matmul::fragment_mma(accumulator, tile, right);
      }
    }
  }

  accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short col, int32_t value) {
    const uint absolute_row = row_base + uint(row);
    if (absolute_row < rows && uint(col) < active_batch) {
      const uint index = uint(col) * rows + absolute_row;
      if constexpr (CS_PASS == 1u) {
        partials[index] = value;
      } else {
        int32_t total = value;
        if constexpr (CS_PASS == 2u) {
          total += partials[index];
        }
        const float gain = as_type<float>(uint(gains_bf16[absolute_row]) << 16u);
        const float weight_scale = float(scales[absolute_row]) * gain * codebook_scale;
        output[index] = bfloat(float(total) * weight_scale * activation_scales[uint(col)]);
      }
    }
    return value;
  });
}

// ---------------------------------------------------------------------------
// CsK: component split restricted to a K'-chunk range [k_begin, k_end) with an
// explicit partials mode (0 = write, 1 = accumulate, 2 = accumulate + epilogue),
// so a leaf can run as (half 0, K-slab 0), (half 1, K-slab 0), (half 0, K-slab 1),
// ... and each slab's codes stay L2-resident between its two passes.
// ---------------------------------------------------------------------------

template <uint COLS, uint ROW_SIMDGROUPS, uint PREFETCH, uint PASS>
static inline void qtip_race_v4_csk(
    device const uchar* codes,
    device const int8_t* codebook_split,
    device const int8_t* activations_half,
    device const float* activation_scales,
    device const half* scales,
    device const ushort* gains_bf16,
    device bfloat* output,
    device int32_t* partials,
    float codebook_scale,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint active_batch,
    uint k_begin,
    uint k_end,
    uint mode,
    uint row_tile,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::OperandFragment<int8_t, 1, 2, Ops>;
  using Accumulator = uzu::matmul::Fragment<int32_t, 1, COLS / 16, Ops>;
  using Format = uzu::matmul::IntegerFormat<8, uzu::matmul::Signedness::Signed>;
  using Right = uzu::matmul::DeviceTensorOperand<Format>;

  const uint row_base = row_tile * (ROW_SIMDGROUPS * 16u) + thread_context.simdgroup_index * 16u;
  const short2 lane_position = Ops::get_position(thread_context.simd_lane_id);
  const uint half_columns = groups * 2u;
  device const char2* table = reinterpret_cast<device const char2*>(codebook_split) + (PASS == 0u ? 0u : 65536u);

  const uint row0 = row_base + uint(lane_position.y);
  const uint row1 = row0 + 8u;
  const bool valid0 = row0 < rows;
  const bool valid1 = row1 < rows;
  device const uchar* codes0 = codes + min(row0, rows - 1u) * bytes_per_row;
  device const uchar* codes1 = codes + min(row1, rows - 1u) * bytes_per_row;
  const uint lane_col = uint(lane_position.x);

  auto gather_tile = [&](uint chunk, thread LeftTile& tile) {
    METAL_PRAGMA_UNROLL
    for (ushort fragment_col = 0; fragment_col < 2; ++fragment_col) {
      const uint g0 = (chunk * 32u + lane_col + uint(fragment_col) * 16u) >> 1u;
      const uint block = g0 >> 4u;
      const uint gib = g0 & 15u;
      thread auto& fragment_values = tile.fragment_at(0, fragment_col);
      device const uchar* seq0 = codes0 + block * 17u;
      device const uchar* seq1 = codes1 + block * 17u;
      char2 v00 = table[qtip_race_v4_state(seq0, gib)];
      char2 v01 = table[qtip_race_v4_state(seq0, gib + 1u)];
      char2 v10 = table[qtip_race_v4_state(seq1, gib)];
      char2 v11 = table[qtip_race_v4_state(seq1, gib + 1u)];
      v00 = valid0 ? v00 : char2(0);
      v01 = valid0 ? v01 : char2(0);
      v10 = valid1 ? v10 : char2(0);
      v11 = valid1 ? v11 : char2(0);
      fragment_values[0] = v00.x; fragment_values[1] = v00.y; fragment_values[2] = v01.x; fragment_values[3] = v01.y;
      fragment_values[4] = v10.x; fragment_values[5] = v10.y; fragment_values[6] = v11.x; fragment_values[7] = v11.y;
    }
  };

  Accumulator accumulator;
  const int row_stride_bytes = int(half_columns);
  LeftTile tile0;
  LeftTile tile1;
  gather_tile(k_begin, tile0);
  if (k_begin + 1u < k_end) {
    gather_tile(k_begin + 1u, tile1);
  }
  for (uint chunk = k_begin; chunk < k_end; chunk += 2u) {
    LeftTile next0;
    LeftTile next1;
    if (chunk + 2u < k_end) {
      gather_tile(chunk + 2u, next0);
    }
    if (chunk + 3u < k_end) {
      gather_tile(chunk + 3u, next1);
    }
    {
      const Right right{activations_half + chunk * 32u, row_stride_bytes};
      if (chunk == k_begin) {
        uzu::matmul::fragment_mm(accumulator, tile0, right);
      } else {
        uzu::matmul::fragment_mma(accumulator, tile0, right);
      }
    }
    if (chunk + 1u < k_end) {
      const Right right{activations_half + (chunk + 1u) * 32u, row_stride_bytes};
      uzu::matmul::fragment_mma(accumulator, tile1, right);
    }
    tile0 = next0;
    tile1 = next1;
  }

  accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short col, int32_t value) {
    const uint absolute_row = row_base + uint(row);
    if (absolute_row < rows && uint(col) < active_batch) {
      const uint index = uint(col) * rows + absolute_row;
      if (mode == 0u) {
        partials[index] = value;
      } else if (mode == 1u) {
        partials[index] += value;
      } else {
        const int32_t total = value + partials[index];
        const float gain = as_type<float>(uint(gains_bf16[absolute_row]) << 16u);
        const float weight_scale = float(scales[absolute_row]) * gain * codebook_scale;
        output[index] = bfloat(float(total) * weight_scale * activation_scales[uint(col)]);
      }
    }
    return value;
  });
}

// ---------------------------------------------------------------------------
// V4 split-table: the Q8 table is stored as two 128 KiB char2 tables
// (components 0,1 at entries [0, 65536) and components 2,3 at [65536, 131072)).
// A block of NB chunks is decoded in two temporally separated passes so each
// pass only touches a 128 KiB table footprint; the MXU then consumes the block.
// ---------------------------------------------------------------------------

template <uint COLS, uint ROW_SIMDGROUPS, uint NB, bool KEEP_STATES>
static inline void qtip_race_v4_split(
    device const uchar* codes,
    device const int8_t* codebook,
    device const int8_t* activations,
    device const float* activation_scales,
    device const half* scales,
    device const ushort* gains_bf16,
    device bfloat* output,
    float codebook_scale,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint active_batch,
    uint row_tile,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::OperandFragment<int8_t, 1, 2, Ops>;
  using Accumulator = uzu::matmul::Fragment<int32_t, 1, COLS / 16, Ops>;
  using Format = uzu::matmul::IntegerFormat<8, uzu::matmul::Signedness::Signed>;
  using Right = uzu::matmul::DeviceTensorOperand<Format>;

  const uint row_base = row_tile * (ROW_SIMDGROUPS * 16u) + thread_context.simdgroup_index * 16u;
  const short2 lane_position = Ops::get_position(thread_context.simd_lane_id);
  const uint columns = groups * 4u;
  const uint chunk_count = columns / 32u;
  device const char2* table_lo = reinterpret_cast<device const char2*>(codebook);
  device const char2* table_hi = table_lo + 65536u;

  const uint row0 = row_base + uint(lane_position.y);
  const uint row1 = row0 + 8u;
  const bool valid0 = row0 < rows;
  const bool valid1 = row1 < rows;
  device const uchar* codes0 = codes + min(row0, rows - 1u) * bytes_per_row;
  device const uchar* codes1 = codes + min(row1, rows - 1u) * bytes_per_row;
  const uint lane_group = uint(lane_position.x) >> 2u;

  auto states_of = [&](uint chunk, thread ushort4& states) {
    const uint block = chunk >> 1u;
    const uint gib = (chunk & 1u) * 8u + lane_group;
    device const uchar* seq0 = codes0 + block * 17u;
    device const uchar* seq1 = codes1 + block * 17u;
    states.x = qtip_race_v4_state(seq0, gib);       // row0, fragment col 0
    states.y = qtip_race_v4_state(seq1, gib);       // row1, fragment col 0
    states.z = qtip_race_v4_state(seq0, gib + 4u);  // row0, fragment col 1
    states.w = qtip_race_v4_state(seq1, gib + 4u);  // row1, fragment col 1
  };

  Accumulator accumulator;
  const int row_stride_bytes = int(columns);
  LeftTile tiles[NB];
  ushort4 states[NB];

  for (uint chunk_base = 0; chunk_base < chunk_count; chunk_base += NB) {
    // pass 1: components 0,1
    METAL_PRAGMA_UNROLL
    for (ushort c = 0; c < NB; ++c) {
      ushort4 st;
      states_of(chunk_base + uint(c), st);
      if constexpr (KEEP_STATES) {
        states[c] = st;
      }
      char2 v0 = table_lo[st.x];
      char2 v1 = table_lo[st.y];
      char2 v2 = table_lo[st.z];
      char2 v3 = table_lo[st.w];
      v0 = valid0 ? v0 : char2(0);
      v2 = valid0 ? v2 : char2(0);
      v1 = valid1 ? v1 : char2(0);
      v3 = valid1 ? v3 : char2(0);
      thread auto& f0 = tiles[c].fragment_at(0, 0);
      thread auto& f1 = tiles[c].fragment_at(0, 1);
      f0[0] = v0.x; f0[1] = v0.y; f0[4] = v1.x; f0[5] = v1.y;
      f1[0] = v2.x; f1[1] = v2.y; f1[4] = v3.x; f1[5] = v3.y;
    }
    // pass 2: components 2,3
    METAL_PRAGMA_UNROLL
    for (ushort c = 0; c < NB; ++c) {
      ushort4 st;
      if constexpr (KEEP_STATES) {
        st = states[c];
      } else {
        states_of(chunk_base + uint(c), st);
      }
      char2 v0 = table_hi[st.x];
      char2 v1 = table_hi[st.y];
      char2 v2 = table_hi[st.z];
      char2 v3 = table_hi[st.w];
      v0 = valid0 ? v0 : char2(0);
      v2 = valid0 ? v2 : char2(0);
      v1 = valid1 ? v1 : char2(0);
      v3 = valid1 ? v3 : char2(0);
      thread auto& f0 = tiles[c].fragment_at(0, 0);
      thread auto& f1 = tiles[c].fragment_at(0, 1);
      f0[2] = v0.x; f0[3] = v0.y; f0[6] = v1.x; f0[7] = v1.y;
      f1[2] = v2.x; f1[3] = v2.y; f1[6] = v3.x; f1[7] = v3.y;
    }
    // MXU
    METAL_PRAGMA_UNROLL
    for (ushort c = 0; c < NB; ++c) {
      const uint chunk = chunk_base + uint(c);
      const Right right{activations + chunk * 32u, row_stride_bytes};
      if (chunk == 0u) {
        uzu::matmul::fragment_mm(accumulator, tiles[c], right);
      } else {
        uzu::matmul::fragment_mma(accumulator, tiles[c], right);
      }
    }
  }

  accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short col, int32_t value) {
    const uint absolute_row = row_base + uint(row);
    if (absolute_row < rows && uint(col) < active_batch) {
      const float gain = as_type<float>(uint(gains_bf16[absolute_row]) << 16u);
      const float weight_scale = float(scales[absolute_row]) * gain * codebook_scale;
      output[uint(col) * rows + absolute_row] =
          bfloat(float(value) * weight_scale * activation_scales[uint(col)]);
    }
    return value;
  });
}

// ---------------------------------------------------------------------------
// B16: two row fragments (32 rows) per SIMDgroup, register activation fragment
// ---------------------------------------------------------------------------

template <uint VECTOR_WIDTH, uint TRANSITION_BITS, uint ROW_SIMDGROUPS, uint PREFETCH>
static inline void qtip_race_b16(
    device const uchar* codes,
    device const int8_t* codebook,
    device const int8_t* activations,
    device const float* activation_scales,
    device const half* scales,
    device const ushort* gains_bf16,
    device bfloat* output,
    float codebook_scale,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint active_batch,
    uint row_tile,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::OperandFragment<int8_t, 2, 2, Ops>;
  using RightTile = uzu::matmul::OperandFragment<int8_t, 2, 1, Ops, uzu::matmul::ReadTranspose>;
  using Accumulator = uzu::matmul::Fragment<int32_t, 2, 1, Ops>;

  const uint row_base = row_tile * (ROW_SIMDGROUPS * 32u) + thread_context.simdgroup_index * 32u;
  const short2 lane_position = Ops::get_position(thread_context.simd_lane_id);
  const uint columns = groups * VECTOR_WIDTH;
  const uint chunk_count = columns / 32u;

  QtipRaceLaneGather<VECTOR_WIDTH, TRANSITION_BITS, 0> lanes[2];
  METAL_PRAGMA_UNROLL
  for (ushort fragment_row = 0; fragment_row < 2; ++fragment_row) {
    const uint row0 = row_base + uint(lane_position.y) + uint(fragment_row) * 16u;
    const uint row1 = row0 + 8u;
    lanes[fragment_row].codes0 = codes + min(row0, rows - 1u) * bytes_per_row;
    lanes[fragment_row].codes1 = codes + min(row1, rows - 1u) * bytes_per_row;
    lanes[fragment_row].codebook = codebook;
    lanes[fragment_row].valid0 = row0 < rows;
    lanes[fragment_row].valid1 = row1 < rows;
    lanes[fragment_row].lane_col = uint(lane_position.x);
  }

  auto gather = [&](uint chunk, thread LeftTile& tile) {
    lanes[0].gather(chunk, tile.fragment_at(0, 0), tile.fragment_at(0, 1));
    lanes[1].gather(chunk, tile.fragment_at(1, 0), tile.fragment_at(1, 1));
  };

  Accumulator accumulator;
  if constexpr (PREFETCH == 0u) {
    for (uint chunk = 0; chunk < chunk_count; ++chunk) {
      LeftTile tile;
      gather(chunk, tile);
      RightTile right_tile;
      right_tile.load_from(
          thread_context.simd_lane_id,
          uzu::matmul::fragment_source(activations + chunk * 32u, int(columns)));
      if (chunk == 0u) {
        uzu::matmul::fragment_mm(accumulator, tile, right_tile);
      } else {
        uzu::matmul::fragment_mma(accumulator, tile, right_tile);
      }
    }
  } else if constexpr (PREFETCH == 1u) {
    LeftTile current;
    gather(0u, current);
    for (uint chunk = 0; chunk < chunk_count; ++chunk) {
      LeftTile next;
      if (chunk + 1u < chunk_count) {
        gather(chunk + 1u, next);
      }
      RightTile right_tile;
      right_tile.load_from(
          thread_context.simd_lane_id,
          uzu::matmul::fragment_source(activations + chunk * 32u, int(columns)));
      if (chunk == 0u) {
        uzu::matmul::fragment_mm(accumulator, current, right_tile);
      } else {
        uzu::matmul::fragment_mma(accumulator, current, right_tile);
      }
      current = next;
    }
  } else {
    LeftTile tile0;
    LeftTile tile1;
    gather(0u, tile0);
    if (chunk_count > 1u) {
      gather(1u, tile1);
    }
    for (uint chunk = 0; chunk < chunk_count; chunk += 2u) {
      LeftTile next0;
      LeftTile next1;
      if (chunk + 2u < chunk_count) {
        gather(chunk + 2u, next0);
      }
      if (chunk + 3u < chunk_count) {
        gather(chunk + 3u, next1);
      }
      {
        RightTile right_tile;
        right_tile.load_from(
            thread_context.simd_lane_id,
            uzu::matmul::fragment_source(activations + chunk * 32u, int(columns)));
        if (chunk == 0u) {
          uzu::matmul::fragment_mm(accumulator, tile0, right_tile);
        } else {
          uzu::matmul::fragment_mma(accumulator, tile0, right_tile);
        }
      }
      if (chunk + 1u < chunk_count) {
        RightTile right_tile;
        right_tile.load_from(
            thread_context.simd_lane_id,
            uzu::matmul::fragment_source(activations + (chunk + 1u) * 32u, int(columns)));
        uzu::matmul::fragment_mma(accumulator, tile1, right_tile);
      }
      tile0 = next0;
      tile1 = next1;
    }
  }

  accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short col, int32_t value) {
    const uint absolute_row = row_base + uint(row);
    if (absolute_row < rows && uint(col) < active_batch) {
      const float gain = as_type<float>(uint(gains_bf16[absolute_row]) << 16u);
      const float weight_scale = float(scales[absolute_row]) * gain * codebook_scale;
      output[uint(col) * rows + absolute_row] =
          bfloat(float(value) * weight_scale * activation_scales[uint(col)]);
    }
    return value;
  });
}

// ---------------------------------------------------------------------------
// Kernel entry points
// ---------------------------------------------------------------------------

#define QTIP_RACE_DT_RF_KERNEL(NAME, COLS, VECTOR_WIDTH, TRANSITION_BITS, ROW_SIMDGROUPS, PREFETCH, DIAG, GATHER_ONLY, ROW_FRAGMENTS) \
KERNEL(NAME)( \
    device const uchar* codes, \
    device const int8_t* codebook, \
    device const int8_t* activations, \
    device const float* activation_scales, \
    device const half* scales, \
    device const ushort* gains_bf16, \
    device bfloat* output, \
    device int32_t* partials, \
    const constant float& codebook_scale, \
    const constant uint& rows, \
    const constant uint& groups, \
    const constant uint& bytes_per_row, \
    const constant uint& active_batch, \
    const uint row_tile GROUPS(rows.div_ceil(ROW_SIMDGROUPS * ROW_FRAGMENTS * 16u)), \
    const uint thread_index THREADS(ROW_SIMDGROUPS * 32u), \
    const ThreadContext thread_context \
) { \
  (void)thread_index; \
  qtip_race_dt<COLS, VECTOR_WIDTH, TRANSITION_BITS, ROW_SIMDGROUPS, PREFETCH, DIAG, GATHER_ONLY, ROW_FRAGMENTS>( \
      codes, codebook, activations, activation_scales, scales, gains_bf16, output, partials, \
      codebook_scale, rows, groups, bytes_per_row, active_batch, row_tile, thread_context); \
}

// two row fragments (32 rows) per SIMDgroup: small-suffix (8/16 padded to 32) candidates
QTIP_RACE_DT_RF_KERNEL(QtipRaceV4R2Pf2Sg2B32, 32, 4, 8, 2, 2, 0, false, 2)
QTIP_RACE_DT_RF_KERNEL(QtipRaceV4L15R2Pf2Sg2B32, 32, 4, 8, 2, 2, 3, false, 2)
QTIP_RACE_DT_RF_KERNEL(QtipRaceV4R2Pf2Sg4B32, 32, 4, 8, 4, 2, 0, false, 2)
QTIP_RACE_DT_RF_KERNEL(QtipRaceK2R2Pf2Sg2B32, 32, 2, 4, 2, 2, 0, false, 2)
QTIP_RACE_DT_RF_KERNEL(QtipRaceK2R2Pf0Sg2B32, 32, 2, 4, 2, 0, 0, false, 2)
QTIP_RACE_DT_RF_KERNEL(QtipRaceK3R2Pf2Sg2B32, 32, 2, 6, 2, 2, 0, false, 2)
QTIP_RACE_DT_RF_KERNEL(QtipRaceK3L15R2Pf2Sg2B32, 32, 2, 6, 2, 2, 3, false, 2)
QTIP_RACE_DT_RF_KERNEL(QtipRaceK2L15R2Pf2Sg2B32, 32, 2, 4, 2, 2, 3, false, 2)
QTIP_RACE_DT_RF_KERNEL(QtipRaceK3R2Pf0Sg2B32, 32, 2, 6, 2, 0, 0, false, 2)

// four row fragments (64 rows) per SIMDgroup: more gathers in flight per chunk

#undef QTIP_RACE_DT_RF_KERNEL

#define QTIP_RACE_DT_KERNEL(NAME, COLS, VECTOR_WIDTH, TRANSITION_BITS, ROW_SIMDGROUPS, PREFETCH, DIAG, GATHER_ONLY) \
KERNEL(NAME)( \
    device const uchar* codes, \
    device const int8_t* codebook, \
    device const int8_t* activations, \
    device const float* activation_scales, \
    device const half* scales, \
    device const ushort* gains_bf16, \
    device bfloat* output, \
    device int32_t* partials, \
    const constant float& codebook_scale, \
    const constant uint& rows, \
    const constant uint& groups, \
    const constant uint& bytes_per_row, \
    const constant uint& active_batch, \
    const uint row_tile GROUPS(rows.div_ceil(ROW_SIMDGROUPS * 16u)), \
    const uint thread_index THREADS(ROW_SIMDGROUPS * 32u), \
    const ThreadContext thread_context \
) { \
  (void)thread_index; \
  qtip_race_dt<COLS, VECTOR_WIDTH, TRANSITION_BITS, ROW_SIMDGROUPS, PREFETCH, DIAG, GATHER_ONLY>( \
      codes, codebook, activations, activation_scales, scales, gains_bf16, output, partials, \
      codebook_scale, rows, groups, bytes_per_row, active_batch, row_tile, thread_context); \
}

// V4 (vector width 4, 8-bit transitions)
QTIP_RACE_DT_KERNEL(QtipRaceV4Pf2Sg4B32, 32, 4, 8, 4, 2, 0, false)
QTIP_RACE_DT_KERNEL(QtipRaceV4Pf2Sg2B32, 32, 4, 8, 2, 2, 0, false)
QTIP_RACE_DT_KERNEL(QtipRaceV4Pf2Sg4B64, 64, 4, 8, 4, 2, 0, false)
QTIP_RACE_DT_KERNEL(QtipRaceV4Pf2Sg2B64, 64, 4, 8, 2, 2, 0, false)
QTIP_RACE_DT_KERNEL(QtipRaceV4Pf2Sg8B64, 64, 4, 8, 8, 2, 0, false)
// V4 diagnostics
// L=15 production variants (32768-state trellis, 128 KiB Q8 table): the decode masks every
// state to 15 bits (DIAG 3) and is otherwise the exact single-pass kernel
QTIP_RACE_DT_KERNEL(QtipRaceV4L15Pf2Sg4B32, 32, 4, 8, 4, 2, 3, false)
QTIP_RACE_DT_KERNEL(QtipRaceV4L15Pf2Sg2B32, 32, 4, 8, 2, 2, 3, false)
QTIP_RACE_DT_KERNEL(QtipRaceV4L15Pf2Sg2B64, 64, 4, 8, 2, 2, 3, false)
QTIP_RACE_DT_KERNEL(QtipRaceV4L15Pf2Sg4B64, 64, 4, 8, 4, 2, 3, false)
QTIP_RACE_DT_KERNEL(QtipRaceV4L15Pf2Sg8B64, 64, 4, 8, 8, 2, 3, false)
QTIP_RACE_DT_KERNEL(QtipRaceV4AntiPf2Sg2B32, 32, 4, 8, 2, 2, 8, false)
QTIP_RACE_DT_KERNEL(QtipRaceV4AntiPf2Sg4B32, 32, 4, 8, 4, 2, 8, false)
QTIP_RACE_DT_KERNEL(QtipRaceV4AntiPf2Sg2B64, 64, 4, 8, 2, 2, 8, false)
QTIP_RACE_DT_KERNEL(QtipRaceV4AntiPf2Sg4B64, 64, 4, 8, 4, 2, 8, false)
QTIP_RACE_DT_KERNEL(QtipRaceV4Sign12Pf2Sg2B32, 32, 4, 8, 2, 2, 9, false)
QTIP_RACE_DT_KERNEL(QtipRaceV4Sign12Pf2Sg4B32, 32, 4, 8, 4, 2, 9, false)
QTIP_RACE_DT_KERNEL(QtipRaceV4Sign12Pf2Sg2B64, 64, 4, 8, 2, 2, 9, false)
QTIP_RACE_DT_KERNEL(QtipRaceV4Sign12Pf2Sg4B64, 64, 4, 8, 4, 2, 9, false)
QTIP_RACE_DT_KERNEL(QtipRaceV4Sign14Pf2Sg2B32, 32, 4, 8, 2, 2, 11, false)
QTIP_RACE_DT_KERNEL(QtipRaceV4Sign14Pf2Sg4B32, 32, 4, 8, 4, 2, 11, false)
QTIP_RACE_DT_KERNEL(QtipRaceV4Sign14Pf2Sg2B64, 64, 4, 8, 2, 2, 11, false)
QTIP_RACE_DT_KERNEL(QtipRaceV4Sign14Pf2Sg4B64, 64, 4, 8, 4, 2, 11, false)
// V4 k=3 (12-bit transitions): L=20 five-symmetry-bit table (DIAG 12) and L=17 two-sign table (DIAG 13)
// V2 two-sign 32 KiB tables (DIAG 14): k3 (6-bit) and k2 (4-bit) connected streams
// V2 plain smaller tables: L=15 (DIAG 3, 64 KiB) and L=14 (DIAG 2, 32 KiB), mask-only decode
QTIP_RACE_DT_KERNEL(QtipRaceK3L15Pf2Sg2B32, 32, 2, 6, 2, 2, 3, false)
QTIP_RACE_DT_KERNEL(QtipRaceK3L15Pf0Sg4B32, 32, 2, 6, 4, 0, 3, false)
QTIP_RACE_DT_KERNEL(QtipRaceK3L15Pf2Sg4B64, 64, 2, 6, 4, 2, 3, false)
QTIP_RACE_DT_KERNEL(QtipRaceK3L15Pf2Sg2B64, 64, 2, 6, 2, 2, 3, false)
QTIP_RACE_DT_KERNEL(QtipRaceK2L15Pf2Sg2B32, 32, 2, 4, 2, 2, 3, false)
QTIP_RACE_DT_KERNEL(QtipRaceK2L15Pf0Sg4B32, 32, 2, 4, 4, 0, 3, false)
QTIP_RACE_DT_KERNEL(QtipRaceK2L15Pf2Sg2B64, 64, 2, 4, 2, 2, 3, false)
// V2 k2 (4-bit transitions)
QTIP_RACE_DT_KERNEL(QtipRaceK2Pf0Sg4B32, 32, 2, 4, 4, 0, 0, false)
QTIP_RACE_DT_KERNEL(QtipRaceK2Pf2Sg4B32, 32, 2, 4, 4, 2, 0, false)
QTIP_RACE_DT_KERNEL(QtipRaceK2Pf2Sg2B32, 32, 2, 4, 2, 2, 0, false)
QTIP_RACE_DT_KERNEL(QtipRaceK2Pf2Sg2B64, 64, 2, 4, 2, 2, 0, false)
// V2 k3 (6-bit transitions)
QTIP_RACE_DT_KERNEL(QtipRaceK3Pf0Sg4B32, 32, 2, 6, 4, 0, 0, false)
QTIP_RACE_DT_KERNEL(QtipRaceK3Pf2Sg4B32, 32, 2, 6, 4, 2, 0, false)
QTIP_RACE_DT_KERNEL(QtipRaceK3Pf2Sg2B32, 32, 2, 6, 2, 2, 0, false)
QTIP_RACE_DT_KERNEL(QtipRaceK3Pf2Sg4B64, 64, 2, 6, 4, 2, 0, false)
QTIP_RACE_DT_KERNEL(QtipRaceK3Pf2Sg2B64, 64, 2, 6, 2, 2, 0, false)
// V2 k3 diagnostics

#undef QTIP_RACE_DT_KERNEL


#define QTIP_RACE_B16T_KERNEL(NAME, VECTOR_WIDTH, TRANSITION_BITS, ROW_FRAGMENTS, ROW_SIMDGROUPS, PREFETCH, HALF) \
KERNEL(NAME)( \
    device const uchar* codes, \
    device const int8_t* codebook, \
    device const int8_t* activations, \
    device const float* activation_scales, \
    device const half* scales, \
    device const ushort* gains_bf16, \
    device bfloat* output, \
    device int32_t* partials, \
    const constant float& codebook_scale, \
    const constant uint& rows, \
    const constant uint& groups, \
    const constant uint& bytes_per_row, \
    const constant uint& active_batch, \
    const uint row_tile GROUPS(rows.div_ceil(ROW_SIMDGROUPS * ROW_FRAGMENTS * 16u)), \
    const uint thread_index THREADS(ROW_SIMDGROUPS * 32u), \
    const ThreadContext thread_context \
) { \
  (void)thread_index; \
  qtip_race_b16t<VECTOR_WIDTH, TRANSITION_BITS, ROW_FRAGMENTS, ROW_SIMDGROUPS, PREFETCH, HALF>( \
      codes, codebook, activations, activation_scales, scales, gains_bf16, output, partials, \
      codebook_scale, rows, groups, bytes_per_row, active_batch, row_tile, thread_context); \
}

QTIP_RACE_B16T_KERNEL(QtipRaceV4T2Pf2Sg2B16, 4, 8, 2, 2, 2, 0)
QTIP_RACE_B16T_KERNEL(QtipRaceV4L15T2Pf0Sg2B16, 4, 8, 2, 2, 0, 3)
QTIP_RACE_B16T_KERNEL(QtipRaceV4L15T2Pf2Sg2B16, 4, 8, 2, 2, 2, 3)
QTIP_RACE_B16T_KERNEL(QtipRaceV4L15T4Pf1Sg2B16, 4, 8, 4, 2, 1, 3)
QTIP_RACE_B16T_KERNEL(QtipRaceV4AntiT2Pf2Sg2B16, 4, 8, 2, 2, 2, 8)
QTIP_RACE_B16T_KERNEL(QtipRaceV4Sign14T2Pf2Sg2B16, 4, 8, 2, 2, 2, 11)
QTIP_RACE_B16T_KERNEL(QtipRaceV4Sign12T2Pf2Sg2B16, 4, 8, 2, 2, 2, 9)
QTIP_RACE_B16T_KERNEL(QtipRaceK3L15T2Pf0Sg2B16, 2, 6, 2, 2, 0, 3)
QTIP_RACE_B16T_KERNEL(QtipRaceK3L15T4Pf0Sg2B16, 2, 6, 4, 2, 0, 3)
QTIP_RACE_B16T_KERNEL(QtipRaceK2L15T2Pf0Sg2B16, 2, 4, 2, 2, 0, 3)
QTIP_RACE_B16T_KERNEL(QtipRaceK2L15T2Pf2Sg2B16, 2, 4, 2, 2, 2, 3)
QTIP_RACE_B16T_KERNEL(QtipRaceK3T2Pf0Sg2B16, 2, 6, 2, 2, 0, 0)
QTIP_RACE_B16T_KERNEL(QtipRaceK3T4Pf0Sg2B16, 2, 6, 4, 2, 0, 0)
QTIP_RACE_B16T_KERNEL(QtipRaceK2T2Pf0Sg2B16, 2, 4, 2, 2, 0, 0)
QTIP_RACE_B16T_KERNEL(QtipRaceK2T2Pf2Sg2B16, 2, 4, 2, 2, 2, 0)


// half-table dual-dispatch variants (V4): pass A gathers states < 32768 and writes int32 partials,
// pass B gathers the rest, adds the partials and applies the epilogue

#undef QTIP_RACE_B16T_KERNEL


// occupancy probes: 16 / 32 SIMDgroups per threadgroup (512 / 1024 threads)


#define QTIP_RACE_V4_CS_KERNEL(NAME, COLS, ROW_SIMDGROUPS, ROW_FRAGMENTS, PREFETCH, PASS) \
KERNEL(NAME)( \
    device const uchar* codes, \
    device const int8_t* codebook_split, \
    device const int8_t* activations_half, \
    device const float* activation_scales, \
    device const half* scales, \
    device const ushort* gains_bf16, \
    device bfloat* output, \
    device int32_t* partials, \
    const constant float& codebook_scale, \
    const constant uint& rows, \
    const constant uint& groups, \
    const constant uint& bytes_per_row, \
    const constant uint& active_batch, \
    const uint row_tile GROUPS(rows.div_ceil(ROW_SIMDGROUPS * ROW_FRAGMENTS * 16u)), \
    const uint thread_index THREADS(ROW_SIMDGROUPS * 32u), \
    const ThreadContext thread_context \
) { \
  (void)thread_index; \
  qtip_race_v4_cs<COLS, ROW_SIMDGROUPS, ROW_FRAGMENTS, PREFETCH, PASS>( \
      codes, codebook_split, activations_half, activation_scales, scales, gains_bf16, output, partials, \
      codebook_scale, rows, groups, bytes_per_row, active_batch, row_tile, thread_context); \
}

QTIP_RACE_V4_CS_KERNEL(QtipRaceV4CsPf2Sg4B32Pass0, 32, 4, 1, 2, 0)
QTIP_RACE_V4_CS_KERNEL(QtipRaceV4CsPf2Sg4B32Pass1, 32, 4, 1, 2, 1)
QTIP_RACE_V4_CS_KERNEL(QtipRaceV4CsR2Pf2Sg2B32Pass0, 32, 2, 2, 2, 0)
QTIP_RACE_V4_CS_KERNEL(QtipRaceV4CsR2Pf2Sg2B32Pass1, 32, 2, 2, 2, 1)
QTIP_RACE_V4_CS_KERNEL(QtipRaceV4CsPf2Sg2B64Pass0, 64, 2, 1, 2, 0)
QTIP_RACE_V4_CS_KERNEL(QtipRaceV4CsPf2Sg2B64Pass1, 64, 2, 1, 2, 1)

#undef QTIP_RACE_V4_CS_KERNEL

#define QTIP_RACE_V4_CS_B16T_KERNEL(NAME, ROW_FRAGMENTS, ROW_SIMDGROUPS, PREFETCH, PASS) \
KERNEL(NAME)( \
    device const uchar* codes, \
    device const int8_t* codebook_split, \
    device const int8_t* activations_half, \
    device const float* activation_scales, \
    device const half* scales, \
    device const ushort* gains_bf16, \
    device bfloat* output, \
    device int32_t* partials, \
    const constant float& codebook_scale, \
    const constant uint& rows, \
    const constant uint& groups, \
    const constant uint& bytes_per_row, \
    const constant uint& active_batch, \
    const uint row_tile GROUPS(rows.div_ceil(ROW_SIMDGROUPS * ROW_FRAGMENTS * 16u)), \
    const uint thread_index THREADS(ROW_SIMDGROUPS * 32u), \
    const ThreadContext thread_context \
) { \
  (void)thread_index; \
  qtip_race_v4_cs_b16t<ROW_FRAGMENTS, ROW_SIMDGROUPS, PREFETCH, PASS>( \
      codes, codebook_split, activations_half, activation_scales, scales, gains_bf16, output, partials, \
      codebook_scale, rows, groups, bytes_per_row, active_batch, row_tile, thread_context); \
}

QTIP_RACE_V4_CS_B16T_KERNEL(QtipRaceV4CsT2Pf0Sg2B16Pass0, 2, 2, 0, 0)
QTIP_RACE_V4_CS_B16T_KERNEL(QtipRaceV4CsT2Pf0Sg2B16Pass1, 2, 2, 0, 1)

#undef QTIP_RACE_V4_CS_B16T_KERNEL


#define QTIP_RACE_AS_KERNEL_B32(NAME, VECTOR_WIDTH, TRANSITION_BITS, ROW_SIMDGROUPS, ROW_FRAGMENTS, CS_PASS, PREFETCH) \
KERNEL(NAME)( \
    device const uchar* codes, \
    device const int8_t* codebook, \
    device const int8_t* activations, \
    device const float* activation_scales, \
    device const half* scales, \
    device const ushort* gains_bf16, \
    device bfloat* output, \
    device int32_t* partials, \
    threadgroup int8_t staging[8704], \
    const constant float& codebook_scale, \
    const constant uint& rows, \
    const constant uint& groups, \
    const constant uint& bytes_per_row, \
    const constant uint& active_batch, \
    const uint row_tile GROUPS(rows.div_ceil(ROW_SIMDGROUPS * ROW_FRAGMENTS * 16u)), \
    const uint thread_index THREADS(ROW_SIMDGROUPS * 32u), \
    const ThreadContext thread_context \
) { \
  qtip_race_as<32, VECTOR_WIDTH, TRANSITION_BITS, ROW_SIMDGROUPS, ROW_FRAGMENTS, CS_PASS, PREFETCH>( \
      codes, codebook, activations, activation_scales, scales, gains_bf16, output, partials, staging, \
      codebook_scale, rows, groups, bytes_per_row, active_batch, row_tile, thread_index, thread_context); \
}

#define QTIP_RACE_AS_KERNEL_B64(NAME, VECTOR_WIDTH, TRANSITION_BITS, ROW_SIMDGROUPS, ROW_FRAGMENTS, CS_PASS, PREFETCH) \
KERNEL(NAME)( \
    device const uchar* codes, \
    device const int8_t* codebook, \
    device const int8_t* activations, \
    device const float* activation_scales, \
    device const half* scales, \
    device const ushort* gains_bf16, \
    device bfloat* output, \
    device int32_t* partials, \
    threadgroup int8_t staging[17408], \
    const constant float& codebook_scale, \
    const constant uint& rows, \
    const constant uint& groups, \
    const constant uint& bytes_per_row, \
    const constant uint& active_batch, \
    const uint row_tile GROUPS(rows.div_ceil(ROW_SIMDGROUPS * ROW_FRAGMENTS * 16u)), \
    const uint thread_index THREADS(ROW_SIMDGROUPS * 32u), \
    const ThreadContext thread_context \
) { \
  qtip_race_as<64, VECTOR_WIDTH, TRANSITION_BITS, ROW_SIMDGROUPS, ROW_FRAGMENTS, CS_PASS, PREFETCH>( \
      codes, codebook, activations, activation_scales, scales, gains_bf16, output, partials, staging, \
      codebook_scale, rows, groups, bytes_per_row, active_batch, row_tile, thread_index, thread_context); \
}

QTIP_RACE_AS_KERNEL_B32(QtipRaceV4AsPf1Sg4B32, 4, 8, 4, 1, 0, 1)
QTIP_RACE_AS_KERNEL_B32(QtipRaceV4AsPf1Sg8B32, 4, 8, 8, 1, 0, 1)
QTIP_RACE_AS_KERNEL_B32(QtipRaceV4AsPf1Sg16B32, 4, 8, 16, 1, 0, 1)
QTIP_RACE_AS_KERNEL_B32(QtipRaceV4AsPf0Sg8B32, 4, 8, 8, 1, 0, 0)
QTIP_RACE_AS_KERNEL_B32(QtipRaceV4AsR2Pf1Sg4B32, 4, 8, 4, 2, 0, 1)
QTIP_RACE_AS_KERNEL_B64(QtipRaceV4AsPf1Sg4B64, 4, 8, 4, 1, 0, 1)
QTIP_RACE_AS_KERNEL_B64(QtipRaceV4AsPf1Sg8B64, 4, 8, 8, 1, 0, 1)
QTIP_RACE_AS_KERNEL_B64(QtipRaceV4AsPf1Sg16B64, 4, 8, 16, 1, 0, 1)
QTIP_RACE_AS_KERNEL_B64(QtipRaceV4AsR2Pf1Sg4B64, 4, 8, 4, 2, 0, 1)
QTIP_RACE_AS_KERNEL_B32(QtipRaceV4AsCsPf1Sg8B32Pass1, 4, 8, 8, 1, 1, 1)
QTIP_RACE_AS_KERNEL_B32(QtipRaceV4AsCsPf1Sg8B32Pass2, 4, 8, 8, 1, 2, 1)
QTIP_RACE_AS_KERNEL_B32(QtipRaceV4AsCsPf1Sg16B32Pass1, 4, 8, 16, 1, 1, 1)
QTIP_RACE_AS_KERNEL_B32(QtipRaceV4AsCsPf1Sg16B32Pass2, 4, 8, 16, 1, 2, 1)
QTIP_RACE_AS_KERNEL_B64(QtipRaceV4AsCsPf1Sg8B64Pass1, 4, 8, 8, 1, 1, 1)
QTIP_RACE_AS_KERNEL_B64(QtipRaceV4AsCsPf1Sg8B64Pass2, 4, 8, 8, 1, 2, 1)
QTIP_RACE_AS_KERNEL_B32(QtipRaceK3AsPf1Sg8B32, 2, 6, 8, 1, 0, 1)
QTIP_RACE_AS_KERNEL_B32(QtipRaceK3AsPf1Sg16B32, 2, 6, 16, 1, 0, 1)
QTIP_RACE_AS_KERNEL_B32(QtipRaceK3AsR2Pf1Sg4B32, 2, 6, 4, 2, 0, 1)
QTIP_RACE_AS_KERNEL_B64(QtipRaceK3AsPf1Sg8B64, 2, 6, 8, 1, 0, 1)
QTIP_RACE_AS_KERNEL_B64(QtipRaceK3AsPf1Sg16B64, 2, 6, 16, 1, 0, 1)
QTIP_RACE_AS_KERNEL_B32(QtipRaceK2AsPf1Sg8B32, 2, 4, 8, 1, 0, 1)
QTIP_RACE_AS_KERNEL_B64(QtipRaceK2AsPf1Sg8B64, 2, 4, 8, 1, 0, 1)

#undef QTIP_RACE_AS_KERNEL_B32
#undef QTIP_RACE_AS_KERNEL_B64


// K-packed activation halves for the component split: lo[t][2g+j] = a[t][4g+j], hi[t][2g+j] = a[t][4g+2+j]

// ---------------------------------------------------------------------------
// SW: staged weights (producer/consumer). NP producer SIMDgroups decode the
// trellis and gather INT8 weight tiles [R_TILE x K_STAGE] into double-buffered
// threadgroup memory while NC consumer SIMDgroups run the MXU on the previous
// stage from threadgroup memory against the device-tensor activation operand.
// One threadgroup barrier per stage; gathers and MXU issue on different
// SIMDgroups, so the two pipelines overlap instead of serializing per SIMDgroup.
// ---------------------------------------------------------------------------
template <uint TRANSITION_BITS>
static inline ushort qtip_race_v2_state(device const uchar* row_codes, uint group) {
  if constexpr (TRANSITION_BITS == 4u) {
    // loader-repacked k2 stream: groups 2j / 2j+1 live in bytes j..j+2
    const uint byte = group >> 1u;
    const uint b0 = uint(row_codes[byte]);
    const uint b1 = uint(row_codes[byte + 1u]);
    const uint b2 = uint(row_codes[byte + 2u]);
    const uint even = (b0 << 8u) | b1;
    const uint odd = ((b0 << 12u) | (b1 << 4u) | (b2 >> 4u)) & 0xFFFFu;
    return ushort((group & 1u) ? odd : even);
  } else {
    const uint bit = group * 6u;
    const uint byte = bit >> 3u;
    const uint shift = bit & 7u;
    const uint window = (uint(row_codes[byte]) << 24u) | (uint(row_codes[byte + 1u]) << 16u) |
        (uint(row_codes[byte + 2u]) << 8u) | uint(row_codes[byte + 3u]);
    return ushort((window >> (16u - shift)) & 0xFFFFu);
  }
}

template <uint COLS, uint VECTOR_WIDTH, uint TRANSITION_BITS, uint DIAG, uint NP, uint NC, uint ROW_FRAGMENTS, uint STAGE_CHUNKS>
static inline void qtip_race_sw(
    device const uchar* codes,
    device const int8_t* codebook,
    device const int8_t* activations,
    device const float* activation_scales,
    device const half* scales,
    device const ushort* gains_bf16,
    device bfloat* output,
    threadgroup int8_t* staging,
    float codebook_scale,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint active_batch,
    uint row_tile,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::OperandFragment<int8_t, ROW_FRAGMENTS, 2, Ops>;
  using Accumulator = uzu::matmul::Fragment<int32_t, ROW_FRAGMENTS, COLS / 16, Ops>;
  using Format = uzu::matmul::IntegerFormat<8, uzu::matmul::Signedness::Signed>;
  using Right = uzu::matmul::DeviceTensorOperand<Format>;
  constexpr uint R_TILE = NC * ROW_FRAGMENTS * 16u;
  constexpr uint K_STAGE = STAGE_CHUNKS * 32u;
  constexpr uint STRIDE = K_STAGE + 16u;  // 16-byte skew keeps the consumer's 8-row char4 loads on distinct banks
  constexpr uint BUFFER = R_TILE * STRIDE;
  constexpr uint GROUPS_PER_STAGE = K_STAGE / VECTOR_WIDTH;
  constexpr uint ITEMS = R_TILE * GROUPS_PER_STAGE;
  constexpr uint PRODUCER_LANES = NP * 32u;

  const uint columns = groups * VECTOR_WIDTH;
  const uint chunk_count = columns / 32u;
  const uint stage_count = (chunk_count + STAGE_CHUNKS - 1u) / STAGE_CHUNKS;
  const uint row_base = row_tile * R_TILE;
  const uint sg = thread_context.simdgroup_index;
  const bool producer = sg < NP;
  const uint lane = thread_context.simd_lane_id;
  const short2 lane_position = Ops::get_position(lane);
  const int row_stride_bytes = int(columns);
  const uint consumer_index = producer ? 0u : (sg - NP);
  const uint local_row_base = consumer_index * ROW_FRAGMENTS * 16u;
  const uint producer_lane = sg * 32u + lane;
  Accumulator accumulator;

  auto produce_item = [&](uint item, uint group_base, threadgroup int8_t* buffer) {
    const uint local_row = item / GROUPS_PER_STAGE;
    const uint g = item - local_row * GROUPS_PER_STAGE;
    const uint row = row_base + local_row;
    const bool valid = row < rows;
    device const uchar* row_codes = codes + min(row, rows - 1u) * bytes_per_row;
    const uint global_group = group_base + g;
    if constexpr (VECTOR_WIDTH == 4u) {
      device const uchar* seq = row_codes + (global_group >> 4u) * 17u;
      ushort state = qtip_race_v4_state(seq, global_group & 15u);
      if constexpr (DIAG == 3u) {
        state &= 0x7FFFu;
      }
      char4 value = reinterpret_cast<device const char4*>(codebook)[state];
      value = valid ? value : char4(0);
      *reinterpret_cast<threadgroup char4*>(buffer + local_row * STRIDE + g * 4u) = value;
    } else {
      const ushort state = qtip_race_v2_state<TRANSITION_BITS>(row_codes, global_group);
      char2 value = reinterpret_cast<device const char2*>(codebook)[state];
      value = valid ? value : char2(0);
      *reinterpret_cast<threadgroup char2*>(buffer + local_row * STRIDE + g * 2u) = value;
    }
  };
  auto produce = [&](uint stage, threadgroup int8_t* buffer) {
    const uint group_base = stage * GROUPS_PER_STAGE;
    // four independent items per iteration keep several gathers in flight per lane
    for (uint item = producer_lane; item < ITEMS; item += 4u * PRODUCER_LANES) {
      produce_item(item, group_base, buffer);
      if (item + PRODUCER_LANES < ITEMS) produce_item(item + PRODUCER_LANES, group_base, buffer);
      if (item + 2u * PRODUCER_LANES < ITEMS) produce_item(item + 2u * PRODUCER_LANES, group_base, buffer);
      if (item + 3u * PRODUCER_LANES < ITEMS) produce_item(item + 3u * PRODUCER_LANES, group_base, buffer);
    }
  };
  auto consume = [&](uint stage, threadgroup const int8_t* buffer) {
    const uint chunk_base = stage * STAGE_CHUNKS;
    for (uint local_chunk = 0; local_chunk < STAGE_CHUNKS; ++local_chunk) {
      const uint chunk = chunk_base + local_chunk;
      if (chunk >= chunk_count) {
        break;
      }
      LeftTile tile;
      METAL_PRAGMA_UNROLL
      for (ushort fragment_row = 0; fragment_row < ROW_FRAGMENTS; ++fragment_row) {
        METAL_PRAGMA_UNROLL
        for (ushort fragment_col = 0; fragment_col < 2; ++fragment_col) {
          thread auto& fragment_values = tile.fragment_at(fragment_row, fragment_col);
          METAL_PRAGMA_UNROLL
          for (ushort local_row = 0; local_row < 2; ++local_row) {
            const uint r = local_row_base + uint(fragment_row) * 16u + uint(lane_position.y) + uint(local_row) * 8u;
            const uint k = local_chunk * 32u + uint(lane_position.x) + uint(fragment_col) * 16u;
            const char4 v = *reinterpret_cast<threadgroup const char4*>(buffer + r * STRIDE + k);
            fragment_values[local_row * 4 + 0] = v.x;
            fragment_values[local_row * 4 + 1] = v.y;
            fragment_values[local_row * 4 + 2] = v.z;
            fragment_values[local_row * 4 + 3] = v.w;
          }
        }
      }
      const Right right{activations + chunk * 32u, row_stride_bytes};
      if (chunk == 0u) {
        uzu::matmul::fragment_mm(accumulator, tile, right);
      } else {
        uzu::matmul::fragment_mma(accumulator, tile, right);
      }
    }
  };

  if (producer) {
    produce(0u, staging);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stage = 0; stage < stage_count; ++stage) {
    if (producer) {
      if (stage + 1u < stage_count) {
        produce(stage + 1u, staging + ((stage + 1u) & 1u) * BUFFER);
      }
    } else {
      consume(stage, staging + (stage & 1u) * BUFFER);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!producer) {
    const uint consumer_row0 = row_base + local_row_base;
    accumulator.map_coords(lane, [&](short row, short col, int32_t value) {
      const uint absolute_row = consumer_row0 + uint(row);
      if (absolute_row < rows && uint(col) < active_batch) {
        const uint index = uint(col) * rows + absolute_row;
        const float gain = as_type<float>(uint(gains_bf16[absolute_row]) << 16u);
        const float weight_scale = float(scales[absolute_row]) * gain * codebook_scale;
        output[index] = bfloat(float(value) * weight_scale * activation_scales[uint(col)]);
      }
      return value;
    });
  }
}

// staging bytes = 2 * R_TILE * (K_STAGE + 16); the DSL needs literal array sizes
#define QTIP_RACE_SW_KERNEL_18432(NAME, COLS, VECTOR_WIDTH, TRANSITION_BITS, DIAG, NP, NC, ROW_FRAGMENTS, STAGE_CHUNKS) \
KERNEL(NAME)( \
    device const uchar* codes, \
    device const int8_t* codebook, \
    device const int8_t* activations, \
    device const float* activation_scales, \
    device const half* scales, \
    device const ushort* gains_bf16, \
    device bfloat* output, \
    device int32_t* partials, \
    const constant float& codebook_scale, \
    const constant uint& rows, \
    const constant uint& groups, \
    const constant uint& bytes_per_row, \
    const constant uint& active_batch, \
    const uint row_tile GROUPS(rows.div_ceil(NC * ROW_FRAGMENTS * 16u)), \
    const uint thread_index THREADS((NP + NC) * 32u), \
    threadgroup int8_t staging[18432], \
    const ThreadContext thread_context \
) { \
  (void)thread_index; \
  (void)partials; \
  qtip_race_sw<COLS, VECTOR_WIDTH, TRANSITION_BITS, DIAG, NP, NC, ROW_FRAGMENTS, STAGE_CHUNKS>( \
      codes, codebook, activations, activation_scales, scales, gains_bf16, output, staging, \
      codebook_scale, rows, groups, bytes_per_row, active_batch, row_tile, thread_context); \
}
#define QTIP_RACE_SW_KERNEL_17408(NAME, COLS, VECTOR_WIDTH, TRANSITION_BITS, DIAG, NP, NC, ROW_FRAGMENTS, STAGE_CHUNKS) \
KERNEL(NAME)( \
    device const uchar* codes, \
    device const int8_t* codebook, \
    device const int8_t* activations, \
    device const float* activation_scales, \
    device const half* scales, \
    device const ushort* gains_bf16, \
    device bfloat* output, \
    device int32_t* partials, \
    const constant float& codebook_scale, \
    const constant uint& rows, \
    const constant uint& groups, \
    const constant uint& bytes_per_row, \
    const constant uint& active_batch, \
    const uint row_tile GROUPS(rows.div_ceil(NC * ROW_FRAGMENTS * 16u)), \
    const uint thread_index THREADS((NP + NC) * 32u), \
    threadgroup int8_t staging[17408], \
    const ThreadContext thread_context \
) { \
  (void)thread_index; \
  (void)partials; \
  qtip_race_sw<COLS, VECTOR_WIDTH, TRANSITION_BITS, DIAG, NP, NC, ROW_FRAGMENTS, STAGE_CHUNKS>( \
      codes, codebook, activations, activation_scales, scales, gains_bf16, output, staging, \
      codebook_scale, rows, groups, bytes_per_row, active_batch, row_tile, thread_context); \
}
#define QTIP_RACE_SW_KERNEL_9216(NAME, COLS, VECTOR_WIDTH, TRANSITION_BITS, DIAG, NP, NC, ROW_FRAGMENTS, STAGE_CHUNKS) \
KERNEL(NAME)( \
    device const uchar* codes, \
    device const int8_t* codebook, \
    device const int8_t* activations, \
    device const float* activation_scales, \
    device const half* scales, \
    device const ushort* gains_bf16, \
    device bfloat* output, \
    device int32_t* partials, \
    const constant float& codebook_scale, \
    const constant uint& rows, \
    const constant uint& groups, \
    const constant uint& bytes_per_row, \
    const constant uint& active_batch, \
    const uint row_tile GROUPS(rows.div_ceil(NC * ROW_FRAGMENTS * 16u)), \
    const uint thread_index THREADS((NP + NC) * 32u), \
    threadgroup int8_t staging[9216], \
    const ThreadContext thread_context \
) { \
  (void)thread_index; \
  (void)partials; \
  qtip_race_sw<COLS, VECTOR_WIDTH, TRANSITION_BITS, DIAG, NP, NC, ROW_FRAGMENTS, STAGE_CHUNKS>( \
      codes, codebook, activations, activation_scales, scales, gains_bf16, output, staging, \
      codebook_scale, rows, groups, bytes_per_row, active_batch, row_tile, thread_context); \
}

// V4 L=15: R_TILE 64 x K_STAGE 128 (18432 B), R_TILE 32 x K_STAGE 256 (17408 B), R_TILE 32 x K_STAGE 128 (9216 B)
QTIP_RACE_SW_KERNEL_18432(QtipRaceV4L15Sw22B32, 32, 4, 8, 3, 2, 2, 2, 4)
QTIP_RACE_SW_KERNEL_18432(QtipRaceV4L15Sw42B32, 32, 4, 8, 3, 4, 2, 2, 4)
QTIP_RACE_SW_KERNEL_17408(QtipRaceV4L15Sw22sB32, 32, 4, 8, 3, 2, 2, 1, 8)
QTIP_RACE_SW_KERNEL_9216(QtipRaceV4L15Sw11B32, 32, 4, 8, 3, 1, 1, 2, 4)
QTIP_RACE_SW_KERNEL_9216(QtipRaceV4L15Sw21B32, 32, 4, 8, 3, 2, 1, 2, 4)
QTIP_RACE_SW_KERNEL_18432(QtipRaceV4L15Sw22B64, 64, 4, 8, 3, 2, 2, 2, 4)
QTIP_RACE_SW_KERNEL_18432(QtipRaceV4L15Sw42B64, 64, 4, 8, 3, 4, 2, 2, 4)
QTIP_RACE_SW_KERNEL_17408(QtipRaceV4L15Sw22sB64, 64, 4, 8, 3, 2, 2, 1, 8)
QTIP_RACE_SW_KERNEL_9216(QtipRaceV4L15Sw11B64, 64, 4, 8, 3, 1, 1, 2, 4)
QTIP_RACE_SW_KERNEL_9216(QtipRaceV4L15Sw21B64, 64, 4, 8, 3, 2, 1, 2, 4)
// V2 k3 (6-bit) and k2 (4-bit) connected streams
QTIP_RACE_SW_KERNEL_18432(QtipRaceK3Sw22B32, 32, 2, 6, 0, 2, 2, 2, 4)
QTIP_RACE_SW_KERNEL_18432(QtipRaceK3Sw42B32, 32, 2, 6, 0, 4, 2, 2, 4)
QTIP_RACE_SW_KERNEL_9216(QtipRaceK3Sw21B32, 32, 2, 6, 0, 2, 1, 2, 4)
QTIP_RACE_SW_KERNEL_18432(QtipRaceK3Sw22B64, 64, 2, 6, 0, 2, 2, 2, 4)
QTIP_RACE_SW_KERNEL_18432(QtipRaceK3Sw42B64, 64, 2, 6, 0, 4, 2, 2, 4)
QTIP_RACE_SW_KERNEL_9216(QtipRaceK3Sw21B64, 64, 2, 6, 0, 2, 1, 2, 4)
QTIP_RACE_SW_KERNEL_18432(QtipRaceK2Sw22B32, 32, 2, 4, 0, 2, 2, 2, 4)
QTIP_RACE_SW_KERNEL_18432(QtipRaceK2Sw42B32, 32, 2, 4, 0, 4, 2, 2, 4)
QTIP_RACE_SW_KERNEL_18432(QtipRaceK2Sw22B64, 64, 2, 4, 0, 2, 2, 2, 4)
QTIP_RACE_SW_KERNEL_18432(QtipRaceK2Sw42B64, 64, 2, 4, 0, 4, 2, 2, 4)

#undef QTIP_RACE_SW_KERNEL_18432
#undef QTIP_RACE_SW_KERNEL_17408
#undef QTIP_RACE_SW_KERNEL_9216


// ---------------------------------------------------------------------------
// BNT: transposed kernels for 32/64 tokens. Tokens are the MXU M operand
// (M_FRAGMENTS x 16, loaded once per chunk into registers and shared by every
// row fragment), the gathered weight rows are the paired N operand read
// transposed. Both MXU operands live in registers, so the MXU phase moves no
// memory and the activation traffic per weight is 1/ROW_FRAGMENTS byte instead
// of the tokens/16 bytes the device-tensor operand streams per row fragment.
// ---------------------------------------------------------------------------
template <uint M_FRAGMENTS, uint VECTOR_WIDTH, uint TRANSITION_BITS, uint ROW_FRAGMENTS, uint ROW_SIMDGROUPS, uint PREFETCH, uint DIAG>
static inline void qtip_race_bnt(
    device const uchar* codes,
    device const int8_t* codebook,
    device const int8_t* activations,
    device const float* activation_scales,
    device const half* scales,
    device const ushort* gains_bf16,
    device bfloat* output,
    float codebook_scale,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint active_batch,
    uint row_tile,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::OperandFragment<int8_t, M_FRAGMENTS, 2, Ops>;                                  // tokens x 32 k
  using RightTile = uzu::matmul::OperandFragment<int8_t, 2, ROW_FRAGMENTS, Ops, uzu::matmul::ReadTranspose>;  // stored [rows x k]
  using Accumulator = uzu::matmul::Fragment<int32_t, M_FRAGMENTS, ROW_FRAGMENTS, Ops>;                        // tokens x rows

  const uint row_base = row_tile * (ROW_SIMDGROUPS * ROW_FRAGMENTS * 16u) +
      thread_context.simdgroup_index * ROW_FRAGMENTS * 16u;
  const short2 lane_position = Ops::get_position(thread_context.simd_lane_id);
  const uint columns = groups * VECTOR_WIDTH;
  const uint chunk_count = columns / 32u;

  QtipRaceLaneGather<VECTOR_WIDTH, TRANSITION_BITS, DIAG> lanes[ROW_FRAGMENTS];
  METAL_PRAGMA_UNROLL
  for (ushort fragment_row = 0; fragment_row < ROW_FRAGMENTS; ++fragment_row) {
    const uint row0 = row_base + uint(lane_position.y) + uint(fragment_row) * 16u;
    const uint row1 = row0 + 8u;
    lanes[fragment_row].codes0 = codes + min(row0, rows - 1u) * bytes_per_row;
    lanes[fragment_row].codes1 = codes + min(row1, rows - 1u) * bytes_per_row;
    lanes[fragment_row].codebook = codebook;
    lanes[fragment_row].valid0 = row0 < rows;
    lanes[fragment_row].valid1 = row1 < rows;
    lanes[fragment_row].lane_col = uint(lane_position.x);
  }
  auto gather_tile = [&](uint chunk, thread RightTile& tile) {
    METAL_PRAGMA_UNROLL
    for (ushort fragment_row = 0; fragment_row < ROW_FRAGMENTS; ++fragment_row) {
      lanes[fragment_row].gather(chunk, tile.fragment_at(fragment_row, 0), tile.fragment_at(fragment_row, 1));
    }
  };
  auto load_left = [&](uint chunk, thread LeftTile& left) {
    left.load_from(
        thread_context.simd_lane_id,
        uzu::matmul::fragment_source(activations + chunk * 32u, int(columns)));
  };

  Accumulator accumulator;
  if constexpr (PREFETCH == 0u) {
    for (uint chunk = 0; chunk < chunk_count; ++chunk) {
      RightTile tile;
      gather_tile(chunk, tile);
      LeftTile left;
      load_left(chunk, left);
      if (chunk == 0u) {
        uzu::matmul::fragment_mm(accumulator, left, tile);
      } else {
        uzu::matmul::fragment_mma(accumulator, left, tile);
      }
    }
  } else if constexpr (PREFETCH == 1u) {
    RightTile current;
    gather_tile(0u, current);
    for (uint chunk = 0; chunk < chunk_count; ++chunk) {
      RightTile next;
      if (chunk + 1u < chunk_count) {
        gather_tile(chunk + 1u, next);
      }
      LeftTile left;
      load_left(chunk, left);
      if (chunk == 0u) {
        uzu::matmul::fragment_mm(accumulator, left, current);
      } else {
        uzu::matmul::fragment_mma(accumulator, left, current);
      }
      current = next;
    }
  } else {
    RightTile tile0;
    RightTile tile1;
    gather_tile(0u, tile0);
    if (chunk_count > 1u) {
      gather_tile(1u, tile1);
    }
    for (uint chunk = 0; chunk < chunk_count; chunk += 2u) {
      RightTile next0;
      RightTile next1;
      if (chunk + 2u < chunk_count) {
        gather_tile(chunk + 2u, next0);
      }
      if (chunk + 3u < chunk_count) {
        gather_tile(chunk + 3u, next1);
      }
      {
        LeftTile left;
        load_left(chunk, left);
        if (chunk == 0u) {
          uzu::matmul::fragment_mm(accumulator, left, tile0);
        } else {
          uzu::matmul::fragment_mma(accumulator, left, tile0);
        }
      }
      if (chunk + 1u < chunk_count) {
        LeftTile left;
        load_left(chunk + 1u, left);
        uzu::matmul::fragment_mma(accumulator, left, tile1);
      }
      tile0 = next0;
      tile1 = next1;
    }
  }

  accumulator.map_coords(thread_context.simd_lane_id, [&](short token, short row, int32_t value) {
    const uint absolute_row = row_base + uint(row);
    if (absolute_row < rows && uint(token) < active_batch) {
      const uint index = uint(token) * rows + absolute_row;
      const float gain = as_type<float>(uint(gains_bf16[absolute_row]) << 16u);
      const float weight_scale = float(scales[absolute_row]) * gain * codebook_scale;
      output[index] = bfloat(float(value) * weight_scale * activation_scales[uint(token)]);
    }
    return value;
  });
}


// V4 L=15 (mask 3): 32 tokens = 2 M fragments, 64 tokens = 4
// V2 k3 / k2


#define QTIP_RACE_AS_L15_KERNEL_B32(NAME, VECTOR_WIDTH, TRANSITION_BITS, ROW_SIMDGROUPS, ROW_FRAGMENTS, CS_PASS, PREFETCH) \
KERNEL(NAME)( \
    device const uchar* codes, \
    device const int8_t* codebook, \
    device const int8_t* activations, \
    device const float* activation_scales, \
    device const half* scales, \
    device const ushort* gains_bf16, \
    device bfloat* output, \
    device int32_t* partials, \
    threadgroup int8_t staging[8704], \
    const constant float& codebook_scale, \
    const constant uint& rows, \
    const constant uint& groups, \
    const constant uint& bytes_per_row, \
    const constant uint& active_batch, \
    const uint row_tile GROUPS(rows.div_ceil(ROW_SIMDGROUPS * ROW_FRAGMENTS * 16u)), \
    const uint thread_index THREADS(ROW_SIMDGROUPS * 32u), \
    const ThreadContext thread_context \
) { \
  qtip_race_as<32, VECTOR_WIDTH, TRANSITION_BITS, ROW_SIMDGROUPS, ROW_FRAGMENTS, CS_PASS, PREFETCH, 3>( \
      codes, codebook, activations, activation_scales, scales, gains_bf16, output, partials, staging, \
      codebook_scale, rows, groups, bytes_per_row, active_batch, row_tile, thread_index, thread_context); \
}
#define QTIP_RACE_AS_L15_KERNEL_B64(NAME, VECTOR_WIDTH, TRANSITION_BITS, ROW_SIMDGROUPS, ROW_FRAGMENTS, CS_PASS, PREFETCH) \
KERNEL(NAME)( \
    device const uchar* codes, \
    device const int8_t* codebook, \
    device const int8_t* activations, \
    device const float* activation_scales, \
    device const half* scales, \
    device const ushort* gains_bf16, \
    device bfloat* output, \
    device int32_t* partials, \
    threadgroup int8_t staging[17408], \
    const constant float& codebook_scale, \
    const constant uint& rows, \
    const constant uint& groups, \
    const constant uint& bytes_per_row, \
    const constant uint& active_batch, \
    const uint row_tile GROUPS(rows.div_ceil(ROW_SIMDGROUPS * ROW_FRAGMENTS * 16u)), \
    const uint thread_index THREADS(ROW_SIMDGROUPS * 32u), \
    const ThreadContext thread_context \
) { \
  qtip_race_as<64, VECTOR_WIDTH, TRANSITION_BITS, ROW_SIMDGROUPS, ROW_FRAGMENTS, CS_PASS, PREFETCH, 3>( \
      codes, codebook, activations, activation_scales, scales, gains_bf16, output, partials, staging, \
      codebook_scale, rows, groups, bytes_per_row, active_batch, row_tile, thread_index, thread_context); \
}

QTIP_RACE_AS_L15_KERNEL_B32(QtipRaceV4L15AsPf1Sg8B32, 4, 8, 8, 1, 0, 1)
QTIP_RACE_AS_L15_KERNEL_B32(QtipRaceV4L15AsPf1Sg16B32, 4, 8, 16, 1, 0, 1)
QTIP_RACE_AS_L15_KERNEL_B32(QtipRaceV4L15AsR2Pf1Sg8B32, 4, 8, 8, 2, 0, 1)
QTIP_RACE_AS_L15_KERNEL_B32(QtipRaceV4L15AsPf2Sg8B32, 4, 8, 8, 1, 0, 2)
QTIP_RACE_AS_L15_KERNEL_B64(QtipRaceV4L15AsPf1Sg8B64, 4, 8, 8, 1, 0, 1)
QTIP_RACE_AS_L15_KERNEL_B64(QtipRaceV4L15AsPf1Sg16B64, 4, 8, 16, 1, 0, 1)
QTIP_RACE_AS_L15_KERNEL_B64(QtipRaceV4L15AsR2Pf1Sg8B64, 4, 8, 8, 2, 0, 1)
QTIP_RACE_AS_L15_KERNEL_B64(QtipRaceV4L15AsPf2Sg8B64, 4, 8, 8, 1, 0, 2)
#define QTIP_RACE_AS_ANTI_KERNEL_B64(NAME, VECTOR_WIDTH, TRANSITION_BITS, ROW_SIMDGROUPS, ROW_FRAGMENTS, CS_PASS, PREFETCH) \
KERNEL(NAME)( \
    device const uchar* codes, \
    device const int8_t* codebook, \
    device const int8_t* activations, \
    device const float* activation_scales, \
    device const half* scales, \
    device const ushort* gains_bf16, \
    device bfloat* output, \
    device int32_t* partials, \
    threadgroup int8_t staging[17408], \
    const constant float& codebook_scale, \
    const constant uint& rows, \
    const constant uint& groups, \
    const constant uint& bytes_per_row, \
    const constant uint& active_batch, \
    const uint row_tile GROUPS(rows.div_ceil(ROW_SIMDGROUPS * ROW_FRAGMENTS * 16u)), \
    const uint thread_index THREADS(ROW_SIMDGROUPS * 32u), \
    const ThreadContext thread_context \
) { \
  qtip_race_as<64, VECTOR_WIDTH, TRANSITION_BITS, ROW_SIMDGROUPS, ROW_FRAGMENTS, CS_PASS, PREFETCH, 8>( \
      codes, codebook, activations, activation_scales, scales, gains_bf16, output, partials, staging, \
      codebook_scale, rows, groups, bytes_per_row, active_batch, row_tile, thread_index, thread_context); \
}
QTIP_RACE_AS_ANTI_KERNEL_B64(QtipRaceV4AntiAsPf1Sg16B64, 4, 8, 16, 1, 0, 1)
#undef QTIP_RACE_AS_ANTI_KERNEL_B64
#define QTIP_RACE_AS_SIGN14_KERNEL_B64(NAME, VECTOR_WIDTH, TRANSITION_BITS, ROW_SIMDGROUPS, ROW_FRAGMENTS, CS_PASS, PREFETCH) \
KERNEL(NAME)( \
    device const uchar* codes, \
    device const int8_t* codebook, \
    device const int8_t* activations, \
    device const float* activation_scales, \
    device const half* scales, \
    device const ushort* gains_bf16, \
    device bfloat* output, \
    device int32_t* partials, \
    threadgroup int8_t staging[17408], \
    const constant float& codebook_scale, \
    const constant uint& rows, \
    const constant uint& groups, \
    const constant uint& bytes_per_row, \
    const constant uint& active_batch, \
    const uint row_tile GROUPS(rows.div_ceil(ROW_SIMDGROUPS * ROW_FRAGMENTS * 16u)), \
    const uint thread_index THREADS(ROW_SIMDGROUPS * 32u), \
    const ThreadContext thread_context \
) { \
  qtip_race_as<64, VECTOR_WIDTH, TRANSITION_BITS, ROW_SIMDGROUPS, ROW_FRAGMENTS, CS_PASS, PREFETCH, 11>( \
      codes, codebook, activations, activation_scales, scales, gains_bf16, output, partials, staging, \
      codebook_scale, rows, groups, bytes_per_row, active_batch, row_tile, thread_index, thread_context); \
}
QTIP_RACE_AS_SIGN14_KERNEL_B64(QtipRaceV4Sign14AsPf1Sg16B64, 4, 8, 16, 1, 0, 1)
#undef QTIP_RACE_AS_SIGN14_KERNEL_B64
#undef QTIP_RACE_AS_L15_KERNEL_B32
#undef QTIP_RACE_AS_L15_KERNEL_B64


// ---------------------------------------------------------------------------
// TG: 16 KiB base table (4096 x 4 INT8) resident in threadgroup memory, state
// bits 12..15 flip components 0..3 (history-driven signs). Prefetch-2 body of
// the device-tensor kernel with the gathers redirected to threadgroup memory.
// ---------------------------------------------------------------------------
template <uint COLS, uint ROW_SIMDGROUPS>
static inline void qtip_race_dt_tg(
    device const uchar* codes,
    device const int8_t* codebook,
    device const int8_t* activations,
    device const float* activation_scales,
    device const half* scales,
    device const ushort* gains_bf16,
    device bfloat* output,
    threadgroup int8_t* table,
    float codebook_scale,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint active_batch,
    uint row_tile,
    uint thread_index,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::OperandFragment<int8_t, 1, 2, Ops>;
  using Accumulator = uzu::matmul::Fragment<int32_t, 1, COLS / 16, Ops>;
  using Format = uzu::matmul::IntegerFormat<8, uzu::matmul::Signedness::Signed>;
  using Right = uzu::matmul::DeviceTensorOperand<Format>;
  constexpr uint THREADS = ROW_SIMDGROUPS * 32u;
  for (uint i = thread_index; i < 16384u / 16u; i += THREADS) {
    *reinterpret_cast<threadgroup uint4*>(table + i * 16u) = *reinterpret_cast<device const uint4*>(codebook + i * 16u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup const char4* table_vectors = reinterpret_cast<threadgroup const char4*>(table);

  const uint row_base = row_tile * (ROW_SIMDGROUPS * 16u) + thread_context.simdgroup_index * 16u;
  const short2 lane_position = Ops::get_position(thread_context.simd_lane_id);
  const uint columns = groups * 4u;
  const uint chunk_count = columns / 32u;
  const uint row0 = row_base + uint(lane_position.y);
  const uint row1 = row0 + 8u;
  device const uchar* codes0 = codes + min(row0, rows - 1u) * bytes_per_row;
  device const uchar* codes1 = codes + min(row1, rows - 1u) * bytes_per_row;
  const bool valid0 = row0 < rows;
  const bool valid1 = row1 < rows;
  const uint lane_col = uint(lane_position.x);
  auto flip = [](char4 v, ushort h) {
    return char4((h & 1u) ? -v.x : v.x, (h & 2u) ? -v.y : v.y, (h & 4u) ? -v.z : v.z, (h & 8u) ? -v.w : v.w);
  };
  auto gather_tile = [&](uint chunk, thread LeftTile& tile) {
    const uint block = chunk >> 1u;
    const uint gib = (chunk & 1u) * 8u + (lane_col >> 2u);
    device const uchar* seq0 = codes0 + block * 17u;
    device const uchar* seq1 = codes1 + block * 17u;
    const ushort s00 = qtip_race_v4_state(seq0, gib);
    const ushort s10 = qtip_race_v4_state(seq1, gib);
    const ushort s01 = qtip_race_v4_state(seq0, gib + 4u);
    const ushort s11 = qtip_race_v4_state(seq1, gib + 4u);
    char4 v00 = flip(table_vectors[s00 & 0x0FFFu], s00 >> 12u);
    char4 v10 = flip(table_vectors[s10 & 0x0FFFu], s10 >> 12u);
    char4 v01 = flip(table_vectors[s01 & 0x0FFFu], s01 >> 12u);
    char4 v11 = flip(table_vectors[s11 & 0x0FFFu], s11 >> 12u);
    v00 = valid0 ? v00 : char4(0); v01 = valid0 ? v01 : char4(0);
    v10 = valid1 ? v10 : char4(0); v11 = valid1 ? v11 : char4(0);
    thread auto& f0 = tile.fragment_at(0, 0);
    thread auto& f1 = tile.fragment_at(0, 1);
    f0[0] = v00.x; f0[1] = v00.y; f0[2] = v00.z; f0[3] = v00.w; f0[4] = v10.x; f0[5] = v10.y; f0[6] = v10.z; f0[7] = v10.w;
    f1[0] = v01.x; f1[1] = v01.y; f1[2] = v01.z; f1[3] = v01.w; f1[4] = v11.x; f1[5] = v11.y; f1[6] = v11.z; f1[7] = v11.w;
  };
  Accumulator accumulator;
  const int row_stride_bytes = int(columns);
  LeftTile tile0;
  LeftTile tile1;
  gather_tile(0u, tile0);
  if (chunk_count > 1u) {
    gather_tile(1u, tile1);
  }
  for (uint chunk = 0; chunk < chunk_count; chunk += 2u) {
    LeftTile next0;
    LeftTile next1;
    if (chunk + 2u < chunk_count) {
      gather_tile(chunk + 2u, next0);
    }
    if (chunk + 3u < chunk_count) {
      gather_tile(chunk + 3u, next1);
    }
    {
      const Right right{activations + chunk * 32u, row_stride_bytes};
      if (chunk == 0u) {
        uzu::matmul::fragment_mm(accumulator, tile0, right);
      } else {
        uzu::matmul::fragment_mma(accumulator, tile0, right);
      }
    }
    if (chunk + 1u < chunk_count) {
      const Right right{activations + (chunk + 1u) * 32u, row_stride_bytes};
      uzu::matmul::fragment_mma(accumulator, tile1, right);
    }
    tile0 = next0;
    tile1 = next1;
  }
  accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short col, int32_t value) {
    const uint absolute_row = row_base + uint(row);
    if (absolute_row < rows && uint(col) < active_batch) {
      const uint index = uint(col) * rows + absolute_row;
      const float gain = as_type<float>(uint(gains_bf16[absolute_row]) << 16u);
      const float weight_scale = float(scales[absolute_row]) * gain * codebook_scale;
      output[index] = bfloat(float(value) * weight_scale * activation_scales[uint(col)]);
    }
    return value;
  });
}


// ---------------------------------------------------------------------------
// V8: one uint2 gather per 8 weights. Stream: 17-byte restart-64 blocks, MSB-first
// [20-bit seed][7 x 16-bit symbols]; the state after group g is the 20-bit window at
// bit 16g (bytes 2g..2g+2 >> 4). Table: 4096 x 8 INT8 (32 KiB) indexed by the low 12
// state bits; bits 12..19 negate components 0..7. Gather lanes fetch one whole
// 16x16 fragment per instruction (lane l -> row l & 15, k-octet l >> 4) and the MXU
// layout (lane = 4 consecutive k of rows y, y+8) is assembled with SIMD shuffles.
// ---------------------------------------------------------------------------
static inline uint2 qtip_race_v8_signs(uint2 v, uint h) {
  const uint m0 = ((h & 1u) ? 0x000000FFu : 0u) | ((h & 2u) ? 0x0000FF00u : 0u) | ((h & 4u) ? 0x00FF0000u : 0u) | ((h & 8u) ? 0xFF000000u : 0u);
  const uint m1 = ((h & 16u) ? 0x000000FFu : 0u) | ((h & 32u) ? 0x0000FF00u : 0u) | ((h & 64u) ? 0x00FF0000u : 0u) | ((h & 128u) ? 0xFF000000u : 0u);
  return uint2(qtip_race_negate_bytes(v.x, m0), qtip_race_negate_bytes(v.y, m1));
}

template <uint COLS, uint ROW_SIMDGROUPS, uint PREFETCH>
static inline void qtip_race_dt_v8(
    device const uchar* codes,
    device const int8_t* codebook,
    device const int8_t* activations,
    device const float* activation_scales,
    device const half* scales,
    device const ushort* gains_bf16,
    device bfloat* output,
    float codebook_scale,
    uint rows,
    uint groups,
    uint bytes_per_row,
    uint active_batch,
    uint row_tile,
    const thread ThreadContext& thread_context) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using LeftTile = uzu::matmul::OperandFragment<int8_t, 1, 2, Ops>;
  using Accumulator = uzu::matmul::Fragment<int32_t, 1, COLS / 16, Ops>;
  using Format = uzu::matmul::IntegerFormat<8, uzu::matmul::Signedness::Signed>;
  using Right = uzu::matmul::DeviceTensorOperand<Format>;
  device const uint2* table = reinterpret_cast<device const uint2*>(codebook);

  const uint row_base = row_tile * (ROW_SIMDGROUPS * 16u) + thread_context.simdgroup_index * 16u;
  const uint lane = thread_context.simd_lane_id;
  const short2 lane_position = Ops::get_position(lane);
  const uint columns = groups * 8u;
  const uint chunk_count = columns / 32u;
  // gather-lane role: row l & 15, k-octet (l >> 4) within each 16-k fragment column
  const uint g_row = row_base + (lane & 15u);
  const uint g_oct = lane >> 4u;
  const bool g_valid = g_row < rows;
  device const uchar* g_codes = codes + min(g_row, rows - 1u) * bytes_per_row;
  // MXU-lane role: rows y, y+8; k = x..x+3 -> source gather lanes and word
  const uint x = uint(lane_position.x);
  const uint y = uint(lane_position.y);
  const ushort src0 = ushort(y + 16u * (x >> 3u));
  const ushort src1 = ushort(y + 8u + 16u * (x >> 3u));
  const bool high_word = ((x >> 2u) & 1u) != 0u;

  auto gather_word = [&](uint chunk, uint fragment_col) {
    const uint group = chunk * 4u + fragment_col * 2u + g_oct;      // global V8 group index
    device const uchar* seq = g_codes + (group >> 3u) * 17u + (group & 7u) * 2u;
    const uint window = (uint(seq[0]) << 16u) | (uint(seq[1]) << 8u) | uint(seq[2]);
    const uint state = (window >> 4u) & 0xFFFFFu;
    uint2 v = table[state & 0xFFFu];
    v = qtip_race_v8_signs(v, state >> 12u);
    return g_valid ? v : uint2(0u);
  };
  auto gather_tile = [&](uint chunk, thread LeftTile& tile) {
    METAL_PRAGMA_UNROLL
    for (ushort fragment_col = 0; fragment_col < 2; ++fragment_col) {
      const uint2 mine = gather_word(chunk, fragment_col);
      // hand the octets to the MXU layout: 2 rows x (low|high word)
      const uint r0_lo = simd_shuffle(mine.x, src0);
      const uint r0_hi = simd_shuffle(mine.y, src0);
      const uint r1_lo = simd_shuffle(mine.x, src1);
      const uint r1_hi = simd_shuffle(mine.y, src1);
      const char4 v0 = as_type<char4>(high_word ? r0_hi : r0_lo);
      const char4 v1 = as_type<char4>(high_word ? r1_hi : r1_lo);
      thread auto& f = tile.fragment_at(0, fragment_col);
      f[0] = v0.x; f[1] = v0.y; f[2] = v0.z; f[3] = v0.w;
      f[4] = v1.x; f[5] = v1.y; f[6] = v1.z; f[7] = v1.w;
    }
  };

  Accumulator accumulator;
  const int row_stride_bytes = int(columns);
  if constexpr (PREFETCH == 0u) {
    for (uint chunk = 0; chunk < chunk_count; ++chunk) {
      LeftTile tile;
      gather_tile(chunk, tile);
      const Right right{activations + chunk * 32u, row_stride_bytes};
      if (chunk == 0u) {
        uzu::matmul::fragment_mm(accumulator, tile, right);
      } else {
        uzu::matmul::fragment_mma(accumulator, tile, right);
      }
    }
  } else {
    LeftTile tile0;
    LeftTile tile1;
    gather_tile(0u, tile0);
    if (chunk_count > 1u) {
      gather_tile(1u, tile1);
    }
    for (uint chunk = 0; chunk < chunk_count; chunk += 2u) {
      LeftTile next0;
      LeftTile next1;
      if (chunk + 2u < chunk_count) {
        gather_tile(chunk + 2u, next0);
      }
      if (chunk + 3u < chunk_count) {
        gather_tile(chunk + 3u, next1);
      }
      {
        const Right right{activations + chunk * 32u, row_stride_bytes};
        if (chunk == 0u) {
          uzu::matmul::fragment_mm(accumulator, tile0, right);
        } else {
          uzu::matmul::fragment_mma(accumulator, tile0, right);
        }
      }
      if (chunk + 1u < chunk_count) {
        const Right right{activations + (chunk + 1u) * 32u, row_stride_bytes};
        uzu::matmul::fragment_mma(accumulator, tile1, right);
      }
      tile0 = next0;
      tile1 = next1;
    }
  }
  accumulator.map_coords(lane, [&](short row, short col, int32_t value) {
    const uint absolute_row = row_base + uint(row);
    if (absolute_row < rows && uint(col) < active_batch) {
      const uint index = uint(col) * rows + absolute_row;
      const float gain = as_type<float>(uint(gains_bf16[absolute_row]) << 16u);
      const float weight_scale = float(scales[absolute_row]) * gain * codebook_scale;
      output[index] = bfloat(float(value) * weight_scale * activation_scales[uint(col)]);
    }
    return value;
  });
}


KERNEL(QtipRacePermuteHalves)(
    device const int8_t* activations,
    device int8_t* lo,
    device int8_t* hi,
    const constant uint& padded_batch,
    const constant uint& columns,
    const uint index AXIS(padded_batch * columns / 4, 256)) {
  const uint half_columns = columns / 2u;
  const uint token = index / (columns / 4u);
  const uint group = index - token * (columns / 4u);
  const char4 values = *reinterpret_cast<device const char4*>(activations + token * columns + group * 4u);
  *reinterpret_cast<device char2*>(lo + token * half_columns + group * 2u) = char2(values.x, values.y);
  *reinterpret_cast<device char2*>(hi + token * half_columns + group * 2u) = char2(values.z, values.w);
}
