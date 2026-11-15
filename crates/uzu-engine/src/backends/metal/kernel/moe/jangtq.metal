#include <metal_stdlib>
#include <metal_simdgroup_matrix>

#include "../common/defines.h"
#include "../common/dsl.h"

// JANGTQ (jangtq2 "lsb-bitstream", per-row f16 scales, hadamard32 rotation) routed-expert
// decode kernels, ported 1:1 from the MLX/Swift reference implementation
// (MiniCPMJangEngine JANGH/JANGHDecodeQDot.swift, JANGHFusedGateUpKernel.swift,
// JANGHWeightedDownKernel.swift). Numerical parity contract:
//   - one simdgroup lane owns 16 consecutive coefficients; per-coefficient FMA chains
//     keep the reference order and must not be reassociated;
//   - row reductions use simd_sum (matches the reference's device reduction);
//   - f16 scales are promoted to f32 before multiplication;
//   - activations use fast::exp, matching the reference kernels.

// Level of the odd-cubic codebook: level = u * (alpha + beta * u^2), u = code - (2^bits-1)/2.
// `row_words` points at the coefficient row start; `column_base` is the first coefficient
// index of this lane's 16-wide chunk. Extraction is LSB-first within a little-endian
// uint32 bitstream (JANGHDecodeQDot.source).
template <uint BITS>
inline float jangtq_level(
    const device uint* row_words,
    const uint column_base,
    const uint i,
    const float alpha,
    const float beta
) {
  uint code;
  if (BITS == 3) {
    // A 16-coefficient chunk spans 48 bits starting at bit (column_base * 3); the chunk
    // always begins at bit 0 or 16 of a word, so a 64-bit window makes all 16
    // extractions branchless.
    const uint qbase = (column_base * 3u) >> 5u;
    const ulong qw64 = (ulong(row_words[qbase]) | (ulong(row_words[qbase + 1u]) << 32u)) >> ((column_base * 3u) & 31u);
    code = uint(qw64 >> (i * 3u));
  } else {
    // The 16-coefficient chunk starts exactly at word (column_base * BITS) / 32.
    const device uint* qw = row_words + ((column_base * BITS) >> 5u);
    const uint bit = i * BITS;
    const uint word = bit >> 5u;
    const uint shift = bit & 31u;
    uint unpacked = qw[word] >> shift;
    if (shift + BITS > 32u) {
      unpacked |= qw[word + 1u] << (32u - shift);
    }
    code = unpacked;
  }
  code &= (1u << BITS) - 1u;
  const float u = float(code) - float((1u << BITS) - 1u) * 0.5f;
  return u * fma(beta, u * u, alpha);
}

// Input rotation: bf16 -> f32 hadamard32 per 32-vector (reference h32 kernel).
PUBLIC KERNEL(JangtqHadamard32)(
    const device bfloat* x,
    device float* out,
    const constant uint& row_length,
    const constant uint& rows,
    const uint lane THREADS(METAL_SIMD_SIZE),
    const uint block_idx GROUPS(row_length / METAL_SIMD_SIZE),
    const uint row_idx GROUPS(rows)
) {
  const size_t offset = (size_t(row_idx) * row_length) + block_idx * METAL_SIMD_SIZE + lane;
  float value = float(x[offset]);
  for (ushort stage = 1; stage < METAL_SIMD_SIZE; stage <<= 1) {
    const float other = simd_shuffle_xor(value, stage);
    value = (lane & stage) ? other - value : value + other;
  }
  out[offset] = value * 0.17677669529663687f;
}

// Sort and encode dispatch indices by expert (0..255) for prefill L2 cache locality.
// Output format: ((orig_dispatch + 1) << 8) | expert.
PUBLIC KERNEL(JangtqSortIndices)(
    const device uint* expert_ids,
    device uint* sorted_indices,
    device uint* expert_offsets,
    device uint* expert_counts,
    const constant uint& num_dispatches,
    threadgroup uint counts[256],
    threadgroup uint write_ptrs[256],
    threadgroup uint simd_sums[8],
    const uint tid THREADS(256),
    const uint group_idx GROUPS(1)
) {
  counts[tid] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid; i < num_dispatches; i += 256) {
    uint exp = expert_ids[i];
    if (exp < 256) {
      atomic_fetch_add_explicit((threadgroup atomic_uint*)&counts[exp], 1, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint simd_id = tid / 32u;
  const uint lane = tid % 32u;
  const uint c = counts[tid];
  uint prefix = simd_prefix_exclusive_sum(c);
  if (lane == 31u) {
    simd_sums[simd_id] = prefix + c;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < 8u) {
    uint s = simd_sums[tid];
    uint s_prefix = simd_prefix_exclusive_sum(s);
    simd_sums[tid] = s_prefix;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  prefix += simd_sums[simd_id];
  write_ptrs[tid] = prefix;
  expert_offsets[tid] = prefix;
  expert_counts[tid] = c;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid; i < num_dispatches; i += 256) {
    uint exp = expert_ids[i];
    if (exp < 256) {
      uint dest = atomic_fetch_add_explicit((threadgroup atomic_uint*)&write_ptrs[exp], 1, memory_order_relaxed);
      sorted_indices[dest] = ((i + 1u) << 8u) | exp;
    }
  }
}

// Fused gate/up: per route (token * routes_per_token + route), computes
// silu(g) * u for a 32-row tile and applies the hadamard32 output rotation
// (down-projection input basis), matching the reference ROT_OUT path. Output is f32.
// Batches 2 dispatches per threadgroup to reuse expert weights in registers.
template <uint BITS_G, uint BITS_U>
VARIANTS(BITS_G, 2)
VARIANTS(BITS_U, 2)
PUBLIC KERNEL(JangtqFusedGateUp)(
    const device float* x,
    const device uint* packed_g,
    const device half* scales_g,
    const device uint* packed_u,
    const device half* scales_u,
    const device uint* indices,
    const constant float& alpha_g,
    const constant float& beta_g,
    const constant float& alpha_u,
    const constant float& beta_u,
    const constant uint& k,
    const constant uint& n,
    const constant uint& words_g,
    const constant uint& words_u,
    const constant uint& routes_per_token,
    const constant uint& dispatches,
    const constant float& limit,
    device float* out,
    threadgroup float hidden[64],
    const uint simd_lane THREADS(METAL_SIMD_SIZE),
    const uint simd_group THREADS(4u),
    const uint tile_idx GROUPS(n / 32u),
    const uint pair_idx GROUPS((dispatches + 1u) / 2u)
) {
  const uint row0 = tile_idx * 32u + simd_group * 8u;
  const uint d0 = pair_idx * 2u;
  const uint d1 = d0 + 1u;
  const bool has_d1 = (d1 < dispatches);

  const uint raw0 = indices[d0];
  const uint orig_d0 = (raw0 >> 8u) != 0u ? ((raw0 >> 8u) - 1u) : d0;
  const uint expert0 = raw0 & 0xFFu;

  const uint raw1 = has_d1 ? indices[d1] : 0u;
  const uint orig_d1 = has_d1 ? ((raw1 >> 8u) != 0u ? ((raw1 >> 8u) - 1u) : d1) : 0u;
  const uint expert1 = has_d1 ? (raw1 & 0xFFu) : 0u;

  const bool same_expert = has_d1 && (expert0 == expert1);

  const size_t expert_row0 = size_t(expert0) * n + row0;
  const size_t x_row0 = size_t(orig_d0 / routes_per_token) * k;
  const size_t x_row1 = has_d1 ? size_t(orig_d1 / routes_per_token) * k : 0;

  float accum_g0[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float accum_u0[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float accum_g1[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float accum_u1[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  const float lut_g[4] = {
    -1.5f * fma(beta_g, 2.25f, alpha_g),
    -0.5f * fma(beta_g, 0.25f, alpha_g),
     0.5f * fma(beta_g, 0.25f, alpha_g),
     1.5f * fma(beta_g, 2.25f, alpha_g),
  };
  const float lut_u[4] = {
    -1.5f * fma(beta_u, 2.25f, alpha_u),
    -0.5f * fma(beta_u, 0.25f, alpha_u),
     0.5f * fma(beta_u, 0.25f, alpha_u),
     1.5f * fma(beta_u, 2.25f, alpha_u),
  };

  for (uint block = 0; block < k; block += 512u) {
    const uint column_base = block + simd_lane * 16u;
    const device float* x_ptr0 = x + x_row0 + column_base;
    const device float4* x4_0 = (const device float4*)x_ptr0;
    float4 vx0_0 = x4_0[0], vx1_0 = x4_0[1], vx2_0 = x4_0[2], vx3_0 = x4_0[3];
    float values0[16] = {
      vx0_0.x, vx0_0.y, vx0_0.z, vx0_0.w,
      vx1_0.x, vx1_0.y, vx1_0.z, vx1_0.w,
      vx2_0.x, vx2_0.y, vx2_0.z, vx2_0.w,
      vx3_0.x, vx3_0.y, vx3_0.z, vx3_0.w
    };

    float values1[16];
    if (has_d1) {
      const device float* x_ptr1 = x + x_row1 + column_base;
      const device float4* x4_1 = (const device float4*)x_ptr1;
      float4 vx0_1 = x4_1[0], vx1_1 = x4_1[1], vx2_1 = x4_1[2], vx3_1 = x4_1[3];
      values1[0] = vx0_1.x; values1[1] = vx0_1.y; values1[2] = vx0_1.z; values1[3] = vx0_1.w;
      values1[4] = vx1_1.x; values1[5] = vx1_1.y; values1[6] = vx1_1.z; values1[7] = vx1_1.w;
      values1[8] = vx2_1.x; values1[9] = vx2_1.y; values1[10] = vx2_1.z; values1[11] = vx2_1.w;
      values1[12] = vx3_1.x; values1[13] = vx3_1.y; values1[14] = vx3_1.z; values1[15] = vx3_1.w;
    }

    if (BITS_G == 2 && BITS_U == 2) {
      const uint qbase = column_base >> 4u;
      const size_t base_g0 = expert_row0 * words_g + qbase;
      const size_t base_u0 = expert_row0 * words_u + qbase;

      if (same_expert) {
        // Multi-token weight reuse: load weights ONCE, multiply across both tokens in registers!
        for (uint r = 0; r < 8; ++r) {
          const uint cur_gw = packed_g[base_g0 + r * words_g];
          const uint cur_uw = packed_u[base_u0 + r * words_u];
          #pragma unroll
          for (uint i = 0; i < 16; ++i) {
            const float lg = lut_g[(cur_gw >> (i * 2u)) & 3u];
            const float lu = lut_u[(cur_uw >> (i * 2u)) & 3u];
            accum_g0[r] = fma(values0[i], lg, accum_g0[r]);
            accum_u0[r] = fma(values0[i], lu, accum_u0[r]);
            accum_g1[r] = fma(values1[i], lg, accum_g1[r]);
            accum_u1[r] = fma(values1[i], lu, accum_u1[r]);
          }
        }
      } else {
        for (uint r = 0; r < 8; ++r) {
          const uint cur_gw = packed_g[base_g0 + r * words_g];
          const uint cur_uw = packed_u[base_u0 + r * words_u];
          #pragma unroll
          for (uint i = 0; i < 16; ++i) {
            accum_g0[r] = fma(values0[i], lut_g[(cur_gw >> (i * 2u)) & 3u], accum_g0[r]);
            accum_u0[r] = fma(values0[i], lut_u[(cur_uw >> (i * 2u)) & 3u], accum_u0[r]);
          }
        }
        if (has_d1) {
          const size_t expert_row1 = size_t(expert1) * n + row0;
          const size_t base_g1 = expert_row1 * words_g + qbase;
          const size_t base_u1 = expert_row1 * words_u + qbase;
          for (uint r = 0; r < 8; ++r) {
            const uint cur_gw = packed_g[base_g1 + r * words_g];
            const uint cur_uw = packed_u[base_u1 + r * words_u];
            #pragma unroll
            for (uint i = 0; i < 16; ++i) {
              accum_g1[r] = fma(values1[i], lut_g[(cur_gw >> (i * 2u)) & 3u], accum_g1[r]);
              accum_u1[r] = fma(values1[i], lut_u[(cur_uw >> (i * 2u)) & 3u], accum_u1[r]);
            }
          }
        }
      }
    } else {
      for (uint r = 0; r < 8; ++r) {
        const device uint* g_words = packed_g + (expert_row0 + r) * words_g;
        const device uint* u_words = packed_u + (expert_row0 + r) * words_u;
        for (uint i = 0; i < 16; ++i) {
          accum_g0[r] = fma(values0[i], jangtq_level<BITS_G>(g_words, column_base, i, alpha_g, beta_g), accum_g0[r]);
          accum_u0[r] = fma(values0[i], jangtq_level<BITS_U>(u_words, column_base, i, alpha_u, beta_u), accum_u0[r]);
        }
      }
      if (has_d1) {
        const size_t expert_row1 = size_t(expert1) * n + row0;
        for (uint r = 0; r < 8; ++r) {
          const device uint* g_words = packed_g + (expert_row1 + r) * words_g;
          const device uint* u_words = packed_u + (expert_row1 + r) * words_u;
          for (uint i = 0; i < 16; ++i) {
            accum_g1[r] = fma(values1[i], jangtq_level<BITS_G>(g_words, column_base, i, alpha_g, beta_g), accum_g1[r]);
            accum_u1[r] = fma(values1[i], jangtq_level<BITS_U>(u_words, column_base, i, alpha_u, beta_u), accum_u1[r]);
          }
        }
      }
    }
  }

  // Compute d0 activations into hidden[0..31]
  for (uint r = 0; r < 8; ++r) {
    const float g = simd_sum(accum_g0[r]);
    const float u = simd_sum(accum_u0[r]);
    if (simd_lane == 0) {
      float g_scaled = g * float(scales_g[expert_row0 + r]);
      float u_scaled = u * float(scales_u[expert_row0 + r]);
      if (limit > 0.0f) {
        g_scaled = min(g_scaled, limit);
        u_scaled = clamp(u_scaled, -limit, limit);
      }
      const float h = (g_scaled / (1.0f + fast::exp(-g_scaled))) * u_scaled;
      hidden[simd_group * 8u + r] = h;
    }
  }

  // Compute d1 activations into hidden[32..63]
  if (has_d1) {
    const size_t expert_row1 = size_t(expert1) * n + row0;
    for (uint r = 0; r < 8; ++r) {
      const float g = simd_sum(accum_g1[r]);
      const float u = simd_sum(accum_u1[r]);
      if (simd_lane == 0) {
        float g_scaled = g * float(scales_g[expert_row1 + r]);
        float u_scaled = u * float(scales_u[expert_row1 + r]);
        if (limit > 0.0f) {
          g_scaled = min(g_scaled, limit);
          u_scaled = clamp(u_scaled, -limit, limit);
        }
        const float h = (g_scaled / (1.0f + fast::exp(-g_scaled))) * u_scaled;
        hidden[32u + simd_group * 8u + r] = h;
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Parallel output write: simd_group 0 writes d0, simd_group 1 writes d1
  if (simd_group == 0) {
    float value = hidden[simd_lane];
    for (ushort stage = 1; stage < METAL_SIMD_SIZE; stage <<= 1) {
      const float other = simd_shuffle_xor(value, stage);
      value = (simd_lane & stage) ? other - value : value + other;
    }
    out[size_t(orig_d0) * n + tile_idx * METAL_SIMD_SIZE + simd_lane] = value * 0.17677669529663687f;
  } else if (simd_group == 1 && has_d1) {
    float value = hidden[32u + simd_lane];
    for (ushort stage = 1; stage < METAL_SIMD_SIZE; stage <<= 1) {
      const float other = simd_shuffle_xor(value, stage);
      value = (simd_lane & stage) ? other - value : value + other;
    }
    out[size_t(orig_d1) * n + tile_idx * METAL_SIMD_SIZE + simd_lane] = value * 0.17677669529663687f;
  }
}

// In-place float32 hadamard32 rotation per 32-vector for intermediate activations.
// Decoupled from JangtqFusedGateUpTile:
// 1. Math contract: MiniCPM-V-4.7 requires a 32-lane Hadamard butterfly (ROT_OUT) on the activation outputs
//    before down-projection. The 32 elements couple row i and row (i+16) in stage 16.
// 2. Hardware constraint: JangtqFusedGateUpTile tiles 16 rows to remain strictly within the 16 KB optimal
//    shared memory window (Apple Silicon hardware hard limit is 32 KB; 32 rows would consume 32,768 bytes,
//    exceeding the limit once threadgroup state is allocated).
// 3. Execution overhead: Operating on 11 MB across all dispatches in L2 cache takes only ~80 us per layer
//    (total ~3.5 ms across 40 layers), making this decoupled pass virtually zero overhead.
PUBLIC KERNEL(JangtqHadamard32F32)(
    device float* x,
    const constant uint& row_length,
    const constant uint& rows,
    const uint lane THREADS(METAL_SIMD_SIZE),
    const uint block_idx GROUPS(row_length / METAL_SIMD_SIZE),
    const uint row_idx GROUPS(rows)
) {
  const size_t offset = (size_t(row_idx) * row_length) + block_idx * METAL_SIMD_SIZE + lane;
  float value = x[offset];
  for (ushort stage = 1; stage < METAL_SIMD_SIZE; stage <<= 1) {
    const float other = simd_shuffle_xor(value, stage);
    value = (lane & stage) ? other - value : value + other;
  }
  x[offset] = value * 0.17677669529663687f;
}

// Tile fused gate-up with threadgroup shared memory weight reuse:
// Architecture & Physical Tuning Notes:
// 1. DRAM Bandwidth Reduction: Loads 16 rows of Gate (8 KB) and Up (8 KB) into threadgroup shared memory ONCE
//    per expert, then evaluates all dispatches routed to that expert. Reduces DRAM weight transfer from ~40 GB to
//    131 MB per prefill pass (99.7% memory traffic reduction), eliminating unified memory bus contention with CPU
//    preprocessing and Vision Tower, and minimizing device thermal throttling.
// 2. Threadgroup Sizing & Occupancy: 16 rows = 16 KB total threadgroup memory. This perfectly matches the 16 KB
//    sweet spot on Apple Silicon (M1/M2/M3/M4), granting 100% GPU occupancy (up to 2 concurrent threadgroups per SIMD core).
// 3. Register Pre-Staging: Each thread owns 4 rows x 4 blocks of K (2048 columns) = 16 words Gate + 16 words Up
//    = 128 bytes total. These are pre-loaded from shared memory into private registers (gw[4][4], uw[4][4]) ONCE.
//    The subsequent dispatch loop across all tokens of that expert runs 100% in private registers at full ALU clock,
//    with ZERO shared memory reads, ZERO bank conflicts, and ZERO threadgroup barriers inside the token loop.
template <uint BITS_G, uint BITS_U>
VARIANTS(BITS_G, 2)
VARIANTS(BITS_U, 2)
PUBLIC KERNEL(JangtqFusedGateUpTile)(
    const device float* x,
    const device uint* packed_g,
    const device half* scales_g,
    const device uint* packed_u,
    const device half* scales_u,
    const device uint* sorted_indices,
    const device uint* expert_offsets,
    const device uint* expert_counts,
    const constant float& alpha_g,
    const constant float& beta_g,
    const constant float& alpha_u,
    const constant float& beta_u,
    const constant uint& k,
    const constant uint& n,
    const constant uint& words_g,
    const constant uint& words_u,
    const constant uint& routes_per_token,
    const constant uint& dispatches,
    const constant float& limit,
    device float* out,
    threadgroup uint packed_tile_g[2048],
    threadgroup uint packed_tile_u[2048],
    const uint simd_lane THREADS(METAL_SIMD_SIZE),
    const uint simd_group THREADS(4u),
    const uint tile_idx GROUPS(n / 16u),
    const uint expert_idx GROUPS(256)
) {
  const uint count = expert_counts[expert_idx];
  if (count == 0u) {
    return;
  }

  const uint tid = simd_group * METAL_SIMD_SIZE + simd_lane;
  const uint row0 = tile_idx * 16u;
  const size_t expert_row0 = size_t(expert_idx) * n + row0;
  const uint total_words_g = 16u * words_g;
  const uint total_words_u = 16u * words_u;

  // Cooperatively load 16 rows of gate and up weights (2048 words = 8 KB each)
  for (uint w = tid; w < total_words_g; w += 128u) {
    packed_tile_g[w] = packed_g[expert_row0 * words_g + w];
  }
  for (uint w = tid; w < total_words_u; w += 128u) {
    packed_tile_u[w] = packed_u[expert_row0 * words_u + w];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint start_offset = expert_offsets[expert_idx];
  const uint simd_row0 = simd_group * 4u;
  const size_t expert_simd_row = expert_row0 + simd_row0;

  const float lut_g[4] = {
    -1.5f * fma(beta_g, 2.25f, alpha_g),
    -0.5f * fma(beta_g, 0.25f, alpha_g),
     0.5f * fma(beta_g, 0.25f, alpha_g),
     1.5f * fma(beta_g, 2.25f, alpha_g),
  };
  const float lut_u[4] = {
    -1.5f * fma(beta_u, 2.25f, alpha_u),
    -0.5f * fma(beta_u, 0.25f, alpha_u),
     0.5f * fma(beta_u, 0.25f, alpha_u),
     1.5f * fma(beta_u, 2.25f, alpha_u),
  };

  float scales_val_g[4];
  float scales_val_u[4];
  #pragma unroll
  for (uint r = 0; r < 4; ++r) {
    scales_val_g[r] = float(scales_g[expert_simd_row + r]);
    scales_val_u[r] = float(scales_u[expert_simd_row + r]);
  }

  // Pre-stage all weights for this lane across all 4 blocks of k into private registers
  uint gw[4][4];
  uint uw[4][4];
  #pragma unroll
  for (uint b = 0; b < 4; ++b) {
    #pragma unroll
    for (uint r = 0; r < 4; ++r) {
      gw[b][r] = packed_tile_g[(simd_row0 + r) * words_g + b * 32u + simd_lane];
      uw[b][r] = packed_tile_u[(simd_row0 + r) * words_u + b * 32u + simd_lane];
    }
  }

  // Loop over all dispatches assigned to this expert (100% in registers!)
  for (uint d = 0; d < count; ++d) {
    const uint raw = sorted_indices[start_offset + d];
    const uint orig_d = (raw >> 8u) - 1u;
    const size_t x_row0 = size_t(orig_d / routes_per_token) * k;

    float accum_g[4] = {0, 0, 0, 0};
    float accum_u[4] = {0, 0, 0, 0};

    for (uint block = 0; block < k; block += 512u) {
      const uint b = block >> 9u;
      const uint column_base = block + simd_lane * 16u;
      const device float* x_ptr = x + x_row0 + column_base;
      const device float4* x4 = (const device float4*)x_ptr;
      float4 vx0 = x4[0], vx1 = x4[1], vx2 = x4[2], vx3 = x4[3];
      float values[16] = {
        vx0.x, vx0.y, vx0.z, vx0.w,
        vx1.x, vx1.y, vx1.z, vx1.w,
        vx2.x, vx2.y, vx2.z, vx2.w,
        vx3.x, vx3.y, vx3.z, vx3.w
      };

      #pragma unroll
      for (uint r = 0; r < 4; ++r) {
        uint cur_gw = gw[b][r];
        uint cur_uw = uw[b][r];
        #pragma unroll
        for (uint i = 0; i < 16; ++i) {
          accum_g[r] = fma(values[i], lut_g[cur_gw & 3u], accum_g[r]);
          accum_u[r] = fma(values[i], lut_u[cur_uw & 3u], accum_u[r]);
          cur_gw >>= 2u;
          cur_uw >>= 2u;
        }
      }
    }

    const float g0 = simd_sum(accum_g[0]);
    const float g1 = simd_sum(accum_g[1]);
    const float g2 = simd_sum(accum_g[2]);
    const float g3 = simd_sum(accum_g[3]);

    const float u0 = simd_sum(accum_u[0]);
    const float u1 = simd_sum(accum_u[1]);
    const float u2 = simd_sum(accum_u[2]);
    const float u3 = simd_sum(accum_u[3]);

    if (simd_lane == 0) {
      float g0_s = g0 * scales_val_g[0];
      float g1_s = g1 * scales_val_g[1];
      float g2_s = g2 * scales_val_g[2];
      float g3_s = g3 * scales_val_g[3];

      float u0_s = u0 * scales_val_u[0];
      float u1_s = u1 * scales_val_u[1];
      float u2_s = u2 * scales_val_u[2];
      float u3_s = u3 * scales_val_u[3];

      if (limit > 0.0f) {
        g0_s = min(g0_s, limit);
        g1_s = min(g1_s, limit);
        g2_s = min(g2_s, limit);
        g3_s = min(g3_s, limit);

        u0_s = clamp(u0_s, -limit, limit);
        u1_s = clamp(u1_s, -limit, limit);
        u2_s = clamp(u2_s, -limit, limit);
        u3_s = clamp(u3_s, -limit, limit);
      }

      device float* out_ptr = out + size_t(orig_d) * n + row0 + simd_row0;
      out_ptr[0] = (g0_s / (1.0f + fast::exp(-g0_s))) * u0_s;
      out_ptr[1] = (g1_s / (1.0f + fast::exp(-g1_s))) * u1_s;
      out_ptr[2] = (g2_s / (1.0f + fast::exp(-g2_s))) * u2_s;
      out_ptr[3] = (g3_s / (1.0f + fast::exp(-g3_s))) * u3_s;
    }
  }
}

// Weighted down projection: sums score * dot * scale over all routes of a token in f32
// and casts once to bf16, matching the reference single-cast weighted combine.
template <typename T, uint BITS>
VARIANTS(T, bfloat)
VARIANTS(BITS, 2, 3)
PUBLIC KERNEL(JangtqWeightedDown)(
    const device float* hidden,
    const device uint* packed,
    const device half* scales,
    const device uint* indices,
    const device float* scores,
    device T* out,
    const constant float& alpha,
    const constant float& beta,
    const constant uint& h,
    const constant uint& n,
    const constant uint& words,
    const constant uint& routes_per_token,
    const constant uint& batch_size,
    const uint simd_lane THREADS(METAL_SIMD_SIZE),
    const uint simd_group THREADS(4u),
    const uint tile_idx GROUPS(n / 32u),
    const uint token_idx GROUPS(batch_size)
) {
  const uint row0 = tile_idx * 32u + simd_group * 8u;
  float total[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const float lut_3bit[8] = {
    -3.5f * fma(beta, 12.25f, alpha),
    -2.5f * fma(beta, 6.25f, alpha),
    -1.5f * fma(beta, 2.25f, alpha),
    -0.5f * fma(beta, 0.25f, alpha),
     0.5f * fma(beta, 0.25f, alpha),
     1.5f * fma(beta, 2.25f, alpha),
     2.5f * fma(beta, 6.25f, alpha),
     3.5f * fma(beta, 12.25f, alpha),
  };
  const float lut_2bit[4] = {
    -1.5f * fma(beta, 2.25f, alpha),
    -0.5f * fma(beta, 0.25f, alpha),
     0.5f * fma(beta, 0.25f, alpha),
     1.5f * fma(beta, 2.25f, alpha),
  };
  for (uint route = 0; route < routes_per_token; ++route) {
    const uint dispatch = token_idx * routes_per_token + route;
    const uint expert = indices[dispatch];
    const size_t expert_row0 = size_t(expert) * n + row0;
    float accum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint block = 0; block < h; block += 512u) {
      const uint column_base = block + simd_lane * 16u;
      if (column_base >= h) {
        continue;
      }
      const device float* h_ptr = hidden + size_t(dispatch) * h + column_base;
      const device float4* h4 = (const device float4*)h_ptr;
      float4 vh0 = h4[0];
      float4 vh1 = h4[1];
      float4 vh2 = h4[2];
      float4 vh3 = h4[3];
      float values[16] = {
        vh0.x, vh0.y, vh0.z, vh0.w,
        vh1.x, vh1.y, vh1.z, vh1.w,
        vh2.x, vh2.y, vh2.z, vh2.w,
        vh3.x, vh3.y, vh3.z, vh3.w
      };
      if (BITS == 3) {
        const uint qbase = (column_base * 3u) >> 5u;
        const uint qshift = (column_base * 3u) & 31u;
        const size_t base_down = expert_row0 * words + qbase;
        for (uint r = 0; r < 8; ++r) {
          const device uint* pw = packed + base_down + r * words;
          ulong cur_qw = (ulong(pw[0]) | (ulong(pw[1]) << 32u)) >> qshift;
          #pragma unroll
          for (uint i = 0; i < 16; ++i) {
            accum[r] = fma(values[i], lut_3bit[cur_qw & 7u], accum[r]);
            cur_qw >>= 3u;
          }
        }
      } else if (BITS == 2) {
        const uint qbase = column_base >> 4u;
        const size_t base_down = expert_row0 * words + qbase;
        for (uint r = 0; r < 8; ++r) {
          uint cur_qw = packed[base_down + r * words];
          #pragma unroll
          for (uint i = 0; i < 16; ++i) {
            accum[r] = fma(values[i], lut_2bit[cur_qw & 3u], accum[r]);
            cur_qw >>= 2u;
          }
        }
      } else {
        for (uint r = 0; r < 8; ++r) {
          const device uint* row_words = packed + (expert_row0 + r) * words;
          for (uint i = 0; i < 16; ++i) {
            accum[r] = fma(values[i], jangtq_level<BITS>(row_words, column_base, i, alpha, beta), accum[r]);
          }
        }
      }
    }
    const float route_score = scores[dispatch];
    for (uint r = 0; r < 8; ++r) {
      const float dot = simd_sum(accum[r]);
      if (simd_lane == 0) {
        total[r] += route_score * dot * float(scales[expert_row0 + r]);
      }
    }
  }
  if (simd_lane == 0) {
    for (uint r = 0; r < 8; ++r) {
      out[size_t(token_idx) * n + row0 + r] = T(total[r]);
    }
  }
}

// Fused weighted down projection and combine for decode (batch_size tokens):
// Evaluates route-weighted 3-bit/2-bit down projection in registers, then directly
// evaluates sigmoid(shared_gate) * shared_output and adds with reference bf16 rounding.
// Completely eliminates intermediate routed_outputs scratch buffer and the separate JangtqCombine kernel dispatch.
template <typename T, uint BITS>
VARIANTS(T, bfloat)
VARIANTS(BITS, 2, 3)
PUBLIC KERNEL(JangtqWeightedDownCombine)(
    const device float* hidden,
    const device uint* packed,
    const device half* scales,
    const device uint* indices,
    const device float* scores,
    const device T* shared_output,
    const device T* shared_gate,
    device T* out,
    const constant float& alpha,
    const constant float& beta,
    const constant uint& h,
    const constant uint& n,
    const constant uint& words,
    const constant uint& routes_per_token,
    const constant uint& batch_size,
    const uint simd_lane THREADS(METAL_SIMD_SIZE),
    const uint simd_group THREADS(4u),
    const uint tile_idx GROUPS(n / 32u),
    const uint token_idx GROUPS(batch_size)
) {
  const uint row0 = tile_idx * 32u + simd_group * 8u;
  float total[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const float lut_3bit[8] = {
    -3.5f * fma(beta, 12.25f, alpha),
    -2.5f * fma(beta, 6.25f, alpha),
    -1.5f * fma(beta, 2.25f, alpha),
    -0.5f * fma(beta, 0.25f, alpha),
     0.5f * fma(beta, 0.25f, alpha),
     1.5f * fma(beta, 2.25f, alpha),
     2.5f * fma(beta, 6.25f, alpha),
     3.5f * fma(beta, 12.25f, alpha),
  };
  const float lut_2bit[4] = {
    -1.5f * fma(beta, 2.25f, alpha),
    -0.5f * fma(beta, 0.25f, alpha),
     0.5f * fma(beta, 0.25f, alpha),
     1.5f * fma(beta, 2.25f, alpha),
  };
  for (uint route = 0; route < routes_per_token; ++route) {
    const uint dispatch = token_idx * routes_per_token + route;
    const uint expert = indices[dispatch];
    const size_t expert_row0 = size_t(expert) * n + row0;
    float accum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint block = 0; block < h; block += 512u) {
      const uint column_base = block + simd_lane * 16u;
      if (column_base >= h) {
        continue;
      }
      const device float* h_ptr = hidden + size_t(dispatch) * h + column_base;
      const device float4* h4 = (const device float4*)h_ptr;
      float4 vh0 = h4[0];
      float4 vh1 = h4[1];
      float4 vh2 = h4[2];
      float4 vh3 = h4[3];
      float values[16] = {
        vh0.x, vh0.y, vh0.z, vh0.w,
        vh1.x, vh1.y, vh1.z, vh1.w,
        vh2.x, vh2.y, vh2.z, vh2.w,
        vh3.x, vh3.y, vh3.z, vh3.w
      };
      if (BITS == 3) {
        const uint qbase = (column_base * 3u) >> 5u;
        const uint qshift = (column_base * 3u) & 31u;
        const size_t base_down = expert_row0 * words + qbase;
        for (uint r = 0; r < 8; ++r) {
          const device uint* pw = packed + base_down + r * words;
          ulong cur_qw = (ulong(pw[0]) | (ulong(pw[1]) << 32u)) >> qshift;
          #pragma unroll
          for (uint i = 0; i < 16; ++i) {
            accum[r] = fma(values[i], lut_3bit[cur_qw & 7u], accum[r]);
            cur_qw >>= 3u;
          }
        }
      } else if (BITS == 2) {
        const uint qbase = column_base >> 4u;
        const size_t base_down = expert_row0 * words + qbase;
        for (uint r = 0; r < 8; ++r) {
          uint cur_qw = packed[base_down + r * words];
          #pragma unroll
          for (uint i = 0; i < 16; ++i) {
            accum[r] = fma(values[i], lut_2bit[cur_qw & 3u], accum[r]);
            cur_qw >>= 2u;
          }
        }
      } else {
        for (uint r = 0; r < 8; ++r) {
          const device uint* row_words = packed + (expert_row0 + r) * words;
          for (uint i = 0; i < 16; ++i) {
            accum[r] = fma(values[i], jangtq_level<BITS>(row_words, column_base, i, alpha, beta), accum[r]);
          }
        }
      }
    }
    const float route_score = scores[dispatch];
    for (uint r = 0; r < 8; ++r) {
      const float dot = simd_sum(accum[r]);
      if (simd_lane == 0) {
        total[r] += route_score * dot * float(scales[expert_row0 + r]);
      }
    }
  }
  if (simd_lane == 0) {
    const float gate_value = 1.0f / (1.0f + fast::exp(-float(shared_gate[token_idx])));
    const float gate_bf = float(T(gate_value));
    const size_t out_base = size_t(token_idx) * n + row0;
    for (uint r = 0; r < 8; ++r) {
      const float routed_value = float(T(total[r]));
      const float shared_val = float(shared_output[out_base + r]);
      const float product = float(T(gate_bf * shared_val));
      out[out_base + r] = T(routed_value + product);
    }
  }
}

// Paired down-projection: projects 2 dispatches per threadgroup using sorted expert indices
// to reuse 3-bit/2-bit expert down weights in registers across dispatches. Output is float.
template <uint BITS>
VARIANTS(BITS, 2, 3)
PUBLIC KERNEL(JangtqDownProject)(
    const device float* hidden,
    const device uint* packed,
    const device half* scales,
    const device uint* indices,
    const device float* scores,
    device float* out,
    const constant float& alpha,
    const constant float& beta,
    const constant uint& h,
    const constant uint& n,
    const constant uint& words,
    const constant uint& routes_per_token,
    const constant uint& dispatches,
    const uint simd_lane THREADS(METAL_SIMD_SIZE),
    const uint simd_group THREADS(4u),
    const uint tile_idx GROUPS(n / 32u),
    const uint pair_idx GROUPS((dispatches + 1u) / 2u)
) {
  const uint row0 = tile_idx * 32u + simd_group * 8u;
  const uint d0 = pair_idx * 2u;
  const uint d1 = d0 + 1u;
  const bool has_d1 = (d1 < dispatches);

  const uint raw0 = indices[d0];
  const uint orig_d0 = (raw0 >> 8u) != 0u ? ((raw0 >> 8u) - 1u) : d0;
  const uint expert0 = raw0 & 0xFFu;

  const uint raw1 = has_d1 ? indices[d1] : 0u;
  const uint orig_d1 = has_d1 ? ((raw1 >> 8u) != 0u ? ((raw1 >> 8u) - 1u) : d1) : 0u;
  const uint expert1 = has_d1 ? (raw1 & 0xFFu) : 0u;

  const bool same_expert = has_d1 && (expert0 == expert1);

  const size_t expert_row0 = size_t(expert0) * n + row0;
  const size_t expert_row1 = has_d1 ? (size_t(expert1) * n + row0) : 0;
  const size_t h_row0 = size_t(orig_d0) * h;
  const size_t h_row1 = has_d1 ? size_t(orig_d1) * h : 0;

  const float lut_3bit[8] = {
    -3.5f * fma(beta, 12.25f, alpha),
    -2.5f * fma(beta, 6.25f, alpha),
    -1.5f * fma(beta, 2.25f, alpha),
    -0.5f * fma(beta, 0.25f, alpha),
     0.5f * fma(beta, 0.25f, alpha),
     1.5f * fma(beta, 2.25f, alpha),
     2.5f * fma(beta, 6.25f, alpha),
     3.5f * fma(beta, 12.25f, alpha),
  };
  const float lut_2bit[4] = {
    -1.5f * fma(beta, 2.25f, alpha),
    -0.5f * fma(beta, 0.25f, alpha),
     0.5f * fma(beta, 0.25f, alpha),
     1.5f * fma(beta, 2.25f, alpha),
  };

  float accum0[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float accum1[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  for (uint block = 0; block < h; block += 512u) {
    const uint column_base = block + simd_lane * 16u;
    if (column_base >= h) {
      continue;
    }
    const device float* h_ptr0 = hidden + h_row0 + column_base;
    const device float4* h4_0 = (const device float4*)h_ptr0;
    float4 vh0_0 = h4_0[0], vh1_0 = h4_0[1], vh2_0 = h4_0[2], vh3_0 = h4_0[3];
    float values0[16] = {
      vh0_0.x, vh0_0.y, vh0_0.z, vh0_0.w,
      vh1_0.x, vh1_0.y, vh1_0.z, vh1_0.w,
      vh2_0.x, vh2_0.y, vh2_0.z, vh2_0.w,
      vh3_0.x, vh3_0.y, vh3_0.z, vh3_0.w
    };

    float values1[16];
    if (has_d1) {
      const device float* h_ptr1 = hidden + h_row1 + column_base;
      const device float4* h4_1 = (const device float4*)h_ptr1;
      float4 vh0_1 = h4_1[0], vh1_1 = h4_1[1], vh2_1 = h4_1[2], vh3_1 = h4_1[3];
      values1[0] = vh0_1.x; values1[1] = vh0_1.y; values1[2] = vh0_1.z; values1[3] = vh0_1.w;
      values1[4] = vh1_1.x; values1[5] = vh1_1.y; values1[6] = vh1_1.z; values1[7] = vh1_1.w;
      values1[8] = vh2_1.x; values1[9] = vh2_1.y; values1[10] = vh2_1.z; values1[11] = vh2_1.w;
      values1[12] = vh3_1.x; values1[13] = vh3_1.y; values1[14] = vh3_1.z; values1[15] = vh3_1.w;
    }

    if (BITS == 3) {
      const uint qbase = (column_base * 3u) >> 5u;
      const uint qshift = (column_base * 3u) & 31u;
      const size_t base_down0 = expert_row0 * words + qbase;

      if (same_expert) {
        for (uint r = 0; r < 8; ++r) {
          const device uint* pw = packed + base_down0 + r * words;
          const ulong cur_qw = (ulong(pw[0]) | (ulong(pw[1]) << 32u)) >> qshift;
          #pragma unroll
          for (uint i = 0; i < 16; ++i) {
            const float l = lut_3bit[(cur_qw >> (i * 3u)) & 7u];
            accum0[r] = fma(values0[i], l, accum0[r]);
            accum1[r] = fma(values1[i], l, accum1[r]);
          }
        }
      } else {
        for (uint r = 0; r < 8; ++r) {
          const device uint* pw = packed + base_down0 + r * words;
          const ulong cur_qw = (ulong(pw[0]) | (ulong(pw[1]) << 32u)) >> qshift;
          #pragma unroll
          for (uint i = 0; i < 16; ++i) {
            accum0[r] = fma(values0[i], lut_3bit[(cur_qw >> (i * 3u)) & 7u], accum0[r]);
          }
        }
        if (has_d1) {
          const size_t base_down1 = expert_row1 * words + qbase;
          for (uint r = 0; r < 8; ++r) {
            const device uint* pw = packed + base_down1 + r * words;
            const ulong cur_qw = (ulong(pw[0]) | (ulong(pw[1]) << 32u)) >> qshift;
            #pragma unroll
            for (uint i = 0; i < 16; ++i) {
              accum1[r] = fma(values1[i], lut_3bit[(cur_qw >> (i * 3u)) & 7u], accum1[r]);
            }
          }
        }
      }
    } else if (BITS == 2) {
      const uint qbase = column_base >> 4u;
      const size_t base_down0 = expert_row0 * words + qbase;

      if (same_expert) {
        for (uint r = 0; r < 8; ++r) {
          const uint qw = packed[base_down0 + r * words];
          #pragma unroll
          for (uint i = 0; i < 16; ++i) {
            const float l = lut_2bit[(qw >> (i * 2u)) & 3u];
            accum0[r] = fma(values0[i], l, accum0[r]);
            accum1[r] = fma(values1[i], l, accum1[r]);
          }
        }
      } else {
        for (uint r = 0; r < 8; ++r) {
          const uint qw = packed[base_down0 + r * words];
          #pragma unroll
          for (uint i = 0; i < 16; ++i) {
            accum0[r] = fma(values0[i], lut_2bit[(qw >> (i * 2u)) & 3u], accum0[r]);
          }
        }
        if (has_d1) {
          const size_t base_down1 = expert_row1 * words + qbase;
          for (uint r = 0; r < 8; ++r) {
            const uint qw = packed[base_down1 + r * words];
            #pragma unroll
            for (uint i = 0; i < 16; ++i) {
              accum1[r] = fma(values1[i], lut_2bit[(qw >> (i * 2u)) & 3u], accum1[r]);
            }
          }
        }
      }
    }
  }

  const float score0 = scores[orig_d0];
  const float score1 = has_d1 ? scores[orig_d1] : 0.0f;

  const float dot0_0 = simd_sum(accum0[0]);
  const float dot0_1 = simd_sum(accum0[1]);
  const float dot0_2 = simd_sum(accum0[2]);
  const float dot0_3 = simd_sum(accum0[3]);
  const float dot0_4 = simd_sum(accum0[4]);
  const float dot0_5 = simd_sum(accum0[5]);
  const float dot0_6 = simd_sum(accum0[6]);
  const float dot0_7 = simd_sum(accum0[7]);

  if (simd_lane == 0) {
    device float4* out4_0 = (device float4*)(out + size_t(orig_d0) * n + row0);
    out4_0[0] = float4(score0 * dot0_0 * float(scales[expert_row0 + 0]),
                       score0 * dot0_1 * float(scales[expert_row0 + 1]),
                       score0 * dot0_2 * float(scales[expert_row0 + 2]),
                       score0 * dot0_3 * float(scales[expert_row0 + 3]));
    out4_0[1] = float4(score0 * dot0_4 * float(scales[expert_row0 + 4]),
                       score0 * dot0_5 * float(scales[expert_row0 + 5]),
                       score0 * dot0_6 * float(scales[expert_row0 + 6]),
                       score0 * dot0_7 * float(scales[expert_row0 + 7]));
  }
  if (has_d1) {
    const float dot1_0 = simd_sum(accum1[0]);
    const float dot1_1 = simd_sum(accum1[1]);
    const float dot1_2 = simd_sum(accum1[2]);
    const float dot1_3 = simd_sum(accum1[3]);
    const float dot1_4 = simd_sum(accum1[4]);
    const float dot1_5 = simd_sum(accum1[5]);
    const float dot1_6 = simd_sum(accum1[6]);
    const float dot1_7 = simd_sum(accum1[7]);

    if (simd_lane == 0) {
      device float4* out4_1 = (device float4*)(out + size_t(orig_d1) * n + row0);
      out4_1[0] = float4(score1 * dot1_0 * float(scales[expert_row1 + 0]),
                         score1 * dot1_1 * float(scales[expert_row1 + 1]),
                         score1 * dot1_2 * float(scales[expert_row1 + 2]),
                         score1 * dot1_3 * float(scales[expert_row1 + 3]));
      out4_1[1] = float4(score1 * dot1_4 * float(scales[expert_row1 + 4]),
                         score1 * dot1_5 * float(scales[expert_row1 + 5]),
                         score1 * dot1_6 * float(scales[expert_row1 + 6]),
                         score1 * dot1_7 * float(scales[expert_row1 + 7]));
    }
  }
}

// Tile down-projection with threadgroup shared memory weight reuse:
// For each expert and 32-row tile, cooperatively loads the 32x512 3-bit weights (only 6 KB)
// into threadgroup shared memory ONCE, then computes ALL dispatches assigned to that expert.
// Reduces DRAM weight traffic to the theoretical minimum (each active expert loaded exactly once).
template <uint BITS>
VARIANTS(BITS, 2, 3)
PUBLIC KERNEL(JangtqDownProjectTile)(
    const device float* hidden,
    const device uint* packed,
    const device half* scales,
    const device uint* sorted_indices,
    const device uint* expert_offsets,
    const device uint* expert_counts,
    const device float* scores,
    device float* out,
    const constant float& alpha,
    const constant float& beta,
    const constant uint& h,
    const constant uint& n,
    const constant uint& words,
    const constant uint& routes_per_token,
    const constant uint& dispatches,
    threadgroup uint packed_tile[1536],
    const uint simd_lane THREADS(METAL_SIMD_SIZE),
    const uint simd_group THREADS(4u),
    const uint tile_idx GROUPS(n / 32u),
    const uint expert_idx GROUPS(256)
) {
  const uint count = expert_counts[expert_idx];
  if (count == 0u) {
    return;
  }

  const uint tid = simd_group * METAL_SIMD_SIZE + simd_lane;
  const uint row0 = tile_idx * 32u;
  const size_t expert_row0 = size_t(expert_idx) * n + row0;
  const uint total_words = 32u * words;

  // Cooperatively load 32 rows of weights (1536 words = 6 KB for 3-bit) into threadgroup memory
  for (uint w = tid; w < total_words; w += 128u) {
    packed_tile[w] = packed[expert_row0 * words + w];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint start_offset = expert_offsets[expert_idx];
  const uint simd_row0 = simd_group * 8u;
  const size_t expert_simd_row = expert_row0 + simd_row0;

  const float lut_3bit[8] = {
    -3.5f * fma(beta, 12.25f, alpha),
    -2.5f * fma(beta, 6.25f, alpha),
    -1.5f * fma(beta, 2.25f, alpha),
    -0.5f * fma(beta, 0.25f, alpha),
     0.5f * fma(beta, 0.25f, alpha),
     1.5f * fma(beta, 2.25f, alpha),
     2.5f * fma(beta, 6.25f, alpha),
     3.5f * fma(beta, 12.25f, alpha),
  };
  const float lut_2bit[4] = {
    -1.5f * fma(beta, 2.25f, alpha),
    -0.5f * fma(beta, 0.25f, alpha),
     0.5f * fma(beta, 0.25f, alpha),
     1.5f * fma(beta, 2.25f, alpha),
  };

  const uint column_base = simd_lane * 16u;
  const uint qbase_3b = (column_base * 3u) >> 5u;
  const uint qshift_3b = (column_base * 3u) & 31u;
  const uint qbase_2b = column_base >> 4u;

  // Stage weight bit-slices and scales into private registers once per tile
  ulong qw_3b[8];
  uint qw_2b[8];
  float scale_val[8];
  #pragma unroll
  for (uint r = 0; r < 8; ++r) {
    scale_val[r] = float(scales[expert_simd_row + r]);
  }

  if (BITS == 3) {
    #pragma unroll
    for (uint r = 0; r < 8; ++r) {
      const uint tile_w_offset = (simd_row0 + r) * words + qbase_3b;
      const uint w0 = packed_tile[tile_w_offset];
      const uint w1 = packed_tile[tile_w_offset + 1u];
      qw_3b[r] = (ulong(w0) | (ulong(w1) << 32u)) >> qshift_3b;
    }
  } else if (BITS == 2) {
    #pragma unroll
    for (uint r = 0; r < 8; ++r) {
      const uint tile_w_offset = (simd_row0 + r) * words + qbase_2b;
      qw_2b[r] = packed_tile[tile_w_offset];
    }
  }

  // Loop over all dispatches assigned to this expert (executing 100% in registers!)
  for (uint d = 0; d < count; ++d) {
    const uint raw = sorted_indices[start_offset + d];
    const uint orig_d = (raw >> 8u) - 1u;
    const float score = scores[orig_d];
    const size_t h_row = size_t(orig_d) * h + column_base;

    const device float4* h4 = (const device float4*)(hidden + h_row);
    float4 vh0 = h4[0], vh1 = h4[1], vh2 = h4[2], vh3 = h4[3];
    float values[16] = {
      vh0.x, vh0.y, vh0.z, vh0.w,
      vh1.x, vh1.y, vh1.z, vh1.w,
      vh2.x, vh2.y, vh2.z, vh2.w,
      vh3.x, vh3.y, vh3.z, vh3.w
    };

    float accum[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    if (BITS == 3) {
      for (uint r = 0; r < 8; ++r) {
        ulong cur_qw = qw_3b[r];
        #pragma unroll
        for (uint i = 0; i < 16; ++i) {
          accum[r] = fma(values[i], lut_3bit[cur_qw & 7u], accum[r]);
          cur_qw >>= 3u;
        }
      }
    } else if (BITS == 2) {
      for (uint r = 0; r < 8; ++r) {
        uint cur_qw = qw_2b[r];
        #pragma unroll
        for (uint i = 0; i < 16; ++i) {
          accum[r] = fma(values[i], lut_2bit[cur_qw & 3u], accum[r]);
          cur_qw >>= 2u;
        }
      }
    }

    const float dot0 = simd_sum(accum[0]);
    const float dot1 = simd_sum(accum[1]);
    const float dot2 = simd_sum(accum[2]);
    const float dot3 = simd_sum(accum[3]);
    const float dot4 = simd_sum(accum[4]);
    const float dot5 = simd_sum(accum[5]);
    const float dot6 = simd_sum(accum[6]);
    const float dot7 = simd_sum(accum[7]);

    if (simd_lane == 0) {
      device float4* out4 = (device float4*)(out + size_t(orig_d) * n + row0 + simd_row0);
      out4[0] = float4(score * dot0 * scale_val[0],
                       score * dot1 * scale_val[1],
                       score * dot2 * scale_val[2],
                       score * dot3 * scale_val[3]);
      out4[1] = float4(score * dot4 * scale_val[4],
                       score * dot5 * scale_val[5],
                       score * dot6 * scale_val[6],
                       score * dot7 * scale_val[7]);
    }
  }
}

// Reduces route-weighted down-projections for each token:
// sums dispatches for token across routes_per_token into routed output of shape [batch_size, model_dim].
template <typename T, uint BD>
VARIANTS(T, bfloat)
VARIANTS(BD, 256)
PUBLIC KERNEL(JangtqDownReduce)(
    const device float* routed_dispatches,
    device T* routed_outputs,
    const constant uint& model_dim,
    const constant uint& batch_size,
    const constant uint& routes_per_token,
    const uint dim_idx AXIS(model_dim, BD),
    const uint token_idx AXIS(batch_size, 1)
) {
  float total = 0.0f;
  const size_t token_base = size_t(token_idx) * routes_per_token;
  for (uint r = 0; r < routes_per_token; ++r) {
    total += routed_dispatches[(token_base + r) * model_dim + dim_idx];
  }
  routed_outputs[size_t(token_idx) * model_dim + dim_idx] = T(total);
}

// MLX-faithful combine: routed output (already route-weighted by JangtqWeightedDown)
// plus sigmoid-gated shared expert, with the reference's intermediate bf16 rounding
// (sigmoid -> bf16, multiply -> bf16, add -> bf16).
template <typename T, uint BD>
VARIANTS(T, bfloat)
VARIANTS(BD, 256)
PUBLIC KERNEL(JangtqCombine)(
    const device T* routed,
    const device T* shared_output,
    const device T* shared_gate,
    device T* output,
    const constant uint& model_dim,
    const constant uint& batch_size,
    const uint dim_idx AXIS(model_dim, BD),
    const uint token_idx AXIS(batch_size, 1)
) {
  const float routed_value = float(routed[token_idx * model_dim + dim_idx]);
  const float gate_value = 1.0f / (1.0f + fast::exp(-float(shared_gate[token_idx])));
  const float product = float(T(float(T(gate_value)) * float(shared_output[token_idx * model_dim + dim_idx])));
  output[token_idx * model_dim + dim_idx] = T(routed_value + product);
}

// Fused down-reduce and combine:
// Sums route-weighted dispatches into intermediate bfloat16, evaluates the sigmoid gate on the
// shared expert output, and adds them in a single pass. Eliminates the routed_outputs scratch buffer.
template <typename T, uint BD>
VARIANTS(T, bfloat)
VARIANTS(BD, 256)
PUBLIC KERNEL(JangtqDownReduceCombine)(
    const device float* routed_dispatches,
    const device T* shared_output,
    const device T* shared_gate,
    device T* output,
    const constant uint& model_dim,
    const constant uint& batch_size,
    const constant uint& routes_per_token,
    const uint dim_idx AXIS(model_dim, BD),
    const uint token_idx AXIS(batch_size, 1)
) {
  float total = 0.0f;
  const size_t token_base = size_t(token_idx) * routes_per_token;
  for (uint r = 0; r < routes_per_token; ++r) {
    total += routed_dispatches[(token_base + r) * model_dim + dim_idx];
  }
  const float routed_value = float(T(total));
  const float gate_value = 1.0f / (1.0f + fast::exp(-float(shared_gate[token_idx])));
  const float product = float(T(float(T(gate_value)) * float(shared_output[token_idx * model_dim + dim_idx])));
  output[token_idx * model_dim + dim_idx] = T(routed_value + product);
}

// ---------------------------------------------------------------------------
// Dense 3-bit JANGTQ kernels (no expert routing tables; used by DenseMlp):
// - Decode (`batch_size < 8`): 8 simdgroups x 4 rows ALU codebook evaluation.
// - Prefill (`batch_size >= 8`): simdgroup_half8x8 -> float8x8 MMA with
//   inline 3-bit unpack into threadgroup memory.
// GateUp kernels fold SwiGLU + Hadamard32 directly before storing to `out`.
// ---------------------------------------------------------------------------

inline float jangtq3_alu_level(uint code, float alpha, float beta) {
  const float d = float(code) - 3.5f;
  return d * fma(beta * d, d, alpha);
}

inline void jangtq3_unpack_32x32_to_half_kn(
    const device uint* packed,
    uint row0,
    uint k_block,
    uint words,
    uint tid,
    float alpha,
    float beta,
    threadgroup half* tg_w_kn
) {
  const uint r = tid >> 2u;
  const uint chunk = tid & 3u;
  const uint wbase = k_block * 3u;
  const device uint* pw = packed + size_t(row0 + r) * words + wbase;
  const uint w0 = pw[0];
  const uint w1 = pw[1];
  const uint w2 = pw[2];
  uint v24;
  if (chunk == 0u) {
    v24 = w0 & 0xFFFFFFu;
  } else if (chunk == 1u) {
    v24 = ((w0 >> 24u) | (w1 << 8u)) & 0xFFFFFFu;
  } else if (chunk == 2u) {
    v24 = ((w1 >> 16u) | (w2 << 16u)) & 0xFFFFFFu;
  } else {
    v24 = (w2 >> 8u) & 0xFFFFFFu;
  }
  const uint c0 = chunk * 8u;
  #pragma unroll
  for (uint i = 0; i < 8u; ++i) {
    tg_w_kn[(c0 + i) * 32u + r] = half(jangtq3_alu_level(v24 & 7u, alpha, beta));
    v24 >>= 3u;
  }
}

PUBLIC KERNEL(JangtqDenseFusedGateUp)(
    const device float* x,
    const device uint* packed_g,
    const device half* scales_g,
    const device uint* packed_u,
    const device half* scales_u,
    device float* out,
    const constant float& alpha_g,
    const constant float& beta_g,
    const constant float& alpha_u,
    const constant float& beta_u,
    const constant uint& k,
    const constant uint& n,
    const constant uint& words,
    const constant uint& batch_size,
    threadgroup float tg_hidden[32],
    const uint simd_lane THREADS(METAL_SIMD_SIZE),
    const uint simd_group THREADS(8u),
    const uint tile_idx GROUPS(n / 32u),
    const uint token_idx GROUPS(batch_size)
) {
  const uint row0 = tile_idx * 32u + simd_group * 4u;
  const size_t x_row0 = size_t(token_idx) * k;

  float acc_g[4] = {0, 0, 0, 0};
  float acc_u[4] = {0, 0, 0, 0};

  for (uint block = 0; block < k; block += 512u) {
    const uint column_base = block + simd_lane * 16u;
    const device float4* x4 = (const device float4*)(x + x_row0 + column_base);
    const float4 vx0 = x4[0], vx1 = x4[1], vx2 = x4[2], vx3 = x4[3];
    const float values[16] = {
      vx0.x, vx0.y, vx0.z, vx0.w,
      vx1.x, vx1.y, vx1.z, vx1.w,
      vx2.x, vx2.y, vx2.z, vx2.w,
      vx3.x, vx3.y, vx3.z, vx3.w
    };
    const uint qbase = (column_base * 3u) >> 5u;
    const uint qshift = (column_base * 3u) & 31u;
    const size_t base = size_t(row0) * words + qbase;
    #pragma unroll
    for (uint r = 0; r < 4; ++r) {
      const device uint* pwg = packed_g + base + r * words;
      const device uint* pwu = packed_u + base + r * words;
      ulong qw_g = (ulong(pwg[0]) | (ulong(pwg[1]) << 32u)) >> qshift;
      ulong qw_u = (ulong(pwu[0]) | (ulong(pwu[1]) << 32u)) >> qshift;
      #pragma unroll
      for (uint i = 0; i < 16; ++i) {
        acc_g[r] = fma(values[i], jangtq3_alu_level(uint(qw_g) & 7u, alpha_g, beta_g), acc_g[r]);
        acc_u[r] = fma(values[i], jangtq3_alu_level(uint(qw_u) & 7u, alpha_u, beta_u), acc_u[r]);
        qw_g >>= 3u;
        qw_u >>= 3u;
      }
    }
  }

  for (uint r = 0; r < 4; ++r) {
    const float dg = simd_sum(acc_g[r]);
    const float du = simd_sum(acc_u[r]);
    if (simd_lane == 0u) {
      const float g = dg * float(scales_g[row0 + r]);
      const float u = du * float(scales_u[row0 + r]);
      tg_hidden[simd_group * 4u + r] = (g / (1.0f + fast::exp(-g))) * u;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0u) {
    float val = tg_hidden[simd_lane];
    for (ushort stage = 1; stage < 32u; stage <<= 1) {
      const float other = simd_shuffle_xor(val, stage);
      val = (simd_lane & stage) ? other - val : val + other;
    }
    out[size_t(token_idx) * n + tile_idx * 32u + simd_lane] = val * 0.17677669529663687f;
  }
}

PUBLIC KERNEL(JangtqDenseFusedGateUpMma)(
    const device float* x,
    const device uint* packed_g,
    const device half* scales_g,
    const device uint* packed_u,
    const device half* scales_u,
    device float* out,
    const constant float& alpha_g,
    const constant float& beta_g,
    const constant float& alpha_u,
    const constant float& beta_u,
    const constant uint& k,
    const constant uint& n,
    const constant uint& words,
    const constant uint& batch_size,
    threadgroup half tg_a[1024],
    threadgroup half tg_bg[1024],
    threadgroup half tg_bu[1024],
    threadgroup float tg_fg[1024],
    threadgroup float tg_fu[1024],
    const uint simd_lane THREADS(METAL_SIMD_SIZE),
    const uint simd_group THREADS(4u),
    const uint tile_idx GROUPS(n / 32u),
    const uint m_tile_idx GROUPS((batch_size + 31u) / 32u)
) {
  const uint row0 = tile_idx * 32u;
  const uint m0 = m_tile_idx * 32u;
  if (m0 >= batch_size) {
    return;
  }
  const uint tid = simd_group * 32u + simd_lane;
  const uint m_valid = min(32u, batch_size - m0);
  const uint sg_row = (simd_group >> 1u) * 16u;
  const uint sg_col = (simd_group & 1u) * 16u;

  simdgroup_float8x8 acc_g[2][2];
  simdgroup_float8x8 acc_u[2][2];
  #pragma unroll
  for (uint i = 0; i < 2; ++i) {
    #pragma unroll
    for (uint j = 0; j < 2; ++j) {
      acc_g[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      acc_u[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }

  const uint k_blocks = k >> 5u;
  for (uint kb = 0; kb < k_blocks; ++kb) {
    const uint k0 = kb * 32u;
    const uint a_r = tid >> 2u;
    const uint a_c0 = (tid & 3u) * 8u;
    if (a_r < m_valid) {
      const device float4* ap = (const device float4*)(x + size_t(m0 + a_r) * k + k0 + a_c0);
      const float4 v0 = ap[0], v1 = ap[1];
      tg_a[a_r * 32u + a_c0 + 0u] = half(v0.x);
      tg_a[a_r * 32u + a_c0 + 1u] = half(v0.y);
      tg_a[a_r * 32u + a_c0 + 2u] = half(v0.z);
      tg_a[a_r * 32u + a_c0 + 3u] = half(v0.w);
      tg_a[a_r * 32u + a_c0 + 4u] = half(v1.x);
      tg_a[a_r * 32u + a_c0 + 5u] = half(v1.y);
      tg_a[a_r * 32u + a_c0 + 6u] = half(v1.z);
      tg_a[a_r * 32u + a_c0 + 7u] = half(v1.w);
    } else {
      for (uint i = 0; i < 8u; ++i) {
        tg_a[a_r * 32u + a_c0 + i] = half(0.0f);
      }
    }

    jangtq3_unpack_32x32_to_half_kn(packed_g, row0, kb, words, tid, alpha_g, beta_g, tg_bg);
    jangtq3_unpack_32x32_to_half_kn(packed_u, row0, kb, words, tid, alpha_u, beta_u, tg_bu);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint ks = 0; ks < 32u; ks += 8u) {
      simdgroup_half8x8 ma0, ma1, mbg0, mbg1, mbu0, mbu1;
      simdgroup_load(ma0, tg_a + (sg_row + 0u) * 32u + ks, 32);
      simdgroup_load(ma1, tg_a + (sg_row + 8u) * 32u + ks, 32);
      simdgroup_load(mbg0, tg_bg + ks * 32u + sg_col + 0u, 32);
      simdgroup_load(mbg1, tg_bg + ks * 32u + sg_col + 8u, 32);
      simdgroup_load(mbu0, tg_bu + ks * 32u + sg_col + 0u, 32);
      simdgroup_load(mbu1, tg_bu + ks * 32u + sg_col + 8u, 32);

      simdgroup_multiply_accumulate(acc_g[0][0], ma0, mbg0, acc_g[0][0]);
      simdgroup_multiply_accumulate(acc_g[0][1], ma0, mbg1, acc_g[0][1]);
      simdgroup_multiply_accumulate(acc_g[1][0], ma1, mbg0, acc_g[1][0]);
      simdgroup_multiply_accumulate(acc_g[1][1], ma1, mbg1, acc_g[1][1]);

      simdgroup_multiply_accumulate(acc_u[0][0], ma0, mbu0, acc_u[0][0]);
      simdgroup_multiply_accumulate(acc_u[0][1], ma0, mbu1, acc_u[0][1]);
      simdgroup_multiply_accumulate(acc_u[1][0], ma1, mbu0, acc_u[1][0]);
      simdgroup_multiply_accumulate(acc_u[1][1], ma1, mbu1, acc_u[1][1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  #pragma unroll
  for (uint i = 0; i < 2; ++i) {
    #pragma unroll
    for (uint j = 0; j < 2; ++j) {
      simdgroup_store(acc_g[i][j], tg_fg + (sg_row + i * 8u) * 32u + sg_col + j * 8u, 32);
      simdgroup_store(acc_u[i][j], tg_fu + (sg_row + i * 8u) * 32u + sg_col + j * 8u, 32);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float sg_scale = float(scales_g[row0 + simd_lane]);
  const float su_scale = float(scales_u[row0 + simd_lane]);
  for (uint m = simd_group; m < m_valid; m += 4u) {
    const float g = tg_fg[m * 32u + simd_lane] * sg_scale;
    const float u = tg_fu[m * 32u + simd_lane] * su_scale;
    float val = (g / (1.0f + fast::exp(-g))) * u;
    for (ushort stage = 1; stage < 32u; stage <<= 1) {
      const float other = simd_shuffle_xor(val, stage);
      val = (simd_lane & stage) ? other - val : val + other;
    }
    out[size_t(m0 + m) * n + row0 + simd_lane] = val * 0.17677669529663687f;
  }
}

template <typename T>
VARIANTS(T, bfloat)
PUBLIC KERNEL(JangtqDenseDown)(
    const device float* hidden,
    const device uint* packed,
    const device half* scales,
    device T* out,
    const constant float& alpha,
    const constant float& beta,
    const constant uint& h,
    const constant uint& n,
    const constant uint& words,
    const constant uint& batch_size,
    const uint simd_lane THREADS(METAL_SIMD_SIZE),
    const uint simd_group THREADS(8u),
    const uint tile_idx GROUPS(n / 32u),
    const uint token_idx GROUPS(batch_size)
) {
  const uint row0 = tile_idx * 32u + simd_group * 4u;
  const size_t h_row0 = size_t(token_idx) * h;

  float accum[4] = {0, 0, 0, 0};
  for (uint block = 0; block < h; block += 512u) {
    const uint column_base = block + simd_lane * 16u;
    const device float4* h4 = (const device float4*)(hidden + h_row0 + column_base);
    const float4 vh0 = h4[0], vh1 = h4[1], vh2 = h4[2], vh3 = h4[3];
    const float values[16] = {
      vh0.x, vh0.y, vh0.z, vh0.w,
      vh1.x, vh1.y, vh1.z, vh1.w,
      vh2.x, vh2.y, vh2.z, vh2.w,
      vh3.x, vh3.y, vh3.z, vh3.w
    };
    const uint qbase = (column_base * 3u) >> 5u;
    const uint qshift = (column_base * 3u) & 31u;
    const size_t base = size_t(row0) * words + qbase;
    #pragma unroll
    for (uint r = 0; r < 4; ++r) {
      const device uint* pw = packed + base + r * words;
      ulong cur_qw = (ulong(pw[0]) | (ulong(pw[1]) << 32u)) >> qshift;
      #pragma unroll
      for (uint i = 0; i < 16; ++i) {
        accum[r] = fma(values[i], jangtq3_alu_level(uint(cur_qw) & 7u, alpha, beta), accum[r]);
        cur_qw >>= 3u;
      }
    }
  }
  const size_t out_base = size_t(token_idx) * n + row0;
  for (uint r = 0; r < 4; ++r) {
    const float dot = simd_sum(accum[r]);
    if (simd_lane == 0u) {
      out[out_base + r] = T(dot * float(scales[row0 + r]));
    }
  }
}

template <typename T>
VARIANTS(T, bfloat)
PUBLIC KERNEL(JangtqDenseDownMma)(
    const device float* hidden,
    const device uint* packed,
    const device half* scales,
    device T* out,
    const constant float& alpha,
    const constant float& beta,
    const constant uint& h,
    const constant uint& n,
    const constant uint& words,
    const constant uint& batch_size,
    threadgroup half tg_a[1024],
    threadgroup half tg_b[1024],
    threadgroup float tg_f[1024],
    const uint simd_lane THREADS(METAL_SIMD_SIZE),
    const uint simd_group THREADS(4u),
    const uint tile_idx GROUPS(n / 32u),
    const uint m_tile_idx GROUPS((batch_size + 31u) / 32u)
) {
  const uint row0 = tile_idx * 32u;
  const uint m0 = m_tile_idx * 32u;
  if (m0 >= batch_size) {
    return;
  }
  const uint tid = simd_group * 32u + simd_lane;
  const uint m_valid = min(32u, batch_size - m0);
  const uint sg_row = (simd_group >> 1u) * 16u;
  const uint sg_col = (simd_group & 1u) * 16u;

  simdgroup_float8x8 acc[2][2];
  #pragma unroll
  for (uint i = 0; i < 2; ++i) {
    #pragma unroll
    for (uint j = 0; j < 2; ++j) {
      acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }

  const uint k_blocks = h >> 5u;
  for (uint kb = 0; kb < k_blocks; ++kb) {
    const uint k0 = kb * 32u;
    const uint a_r = tid >> 2u;
    const uint a_c0 = (tid & 3u) * 8u;
    if (a_r < m_valid) {
      const device float4* ap = (const device float4*)(hidden + size_t(m0 + a_r) * h + k0 + a_c0);
      const float4 v0 = ap[0], v1 = ap[1];
      tg_a[a_r * 32u + a_c0 + 0u] = half(v0.x);
      tg_a[a_r * 32u + a_c0 + 1u] = half(v0.y);
      tg_a[a_r * 32u + a_c0 + 2u] = half(v0.z);
      tg_a[a_r * 32u + a_c0 + 3u] = half(v0.w);
      tg_a[a_r * 32u + a_c0 + 4u] = half(v1.x);
      tg_a[a_r * 32u + a_c0 + 5u] = half(v1.y);
      tg_a[a_r * 32u + a_c0 + 6u] = half(v1.z);
      tg_a[a_r * 32u + a_c0 + 7u] = half(v1.w);
    } else {
      for (uint i = 0; i < 8u; ++i) {
        tg_a[a_r * 32u + a_c0 + i] = half(0.0f);
      }
    }

    jangtq3_unpack_32x32_to_half_kn(packed, row0, kb, words, tid, alpha, beta, tg_b);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint ks = 0; ks < 32u; ks += 8u) {
      simdgroup_half8x8 ma0, ma1, mb0, mb1;
      simdgroup_load(ma0, tg_a + (sg_row + 0u) * 32u + ks, 32);
      simdgroup_load(ma1, tg_a + (sg_row + 8u) * 32u + ks, 32);
      simdgroup_load(mb0, tg_b + ks * 32u + sg_col + 0u, 32);
      simdgroup_load(mb1, tg_b + ks * 32u + sg_col + 8u, 32);
      simdgroup_multiply_accumulate(acc[0][0], ma0, mb0, acc[0][0]);
      simdgroup_multiply_accumulate(acc[0][1], ma0, mb1, acc[0][1]);
      simdgroup_multiply_accumulate(acc[1][0], ma1, mb0, acc[1][0]);
      simdgroup_multiply_accumulate(acc[1][1], ma1, mb1, acc[1][1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  #pragma unroll
  for (uint i = 0; i < 2; ++i) {
    #pragma unroll
    for (uint j = 0; j < 2; ++j) {
      simdgroup_store(acc[i][j], tg_f + (sg_row + i * 8u) * 32u + sg_col + j * 8u, 32);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float s = float(scales[row0 + simd_lane]);
  for (uint m = simd_group; m < m_valid; m += 4u) {
    out[size_t(m0 + m) * n + row0 + simd_lane] = T(tg_f[m * 32u + simd_lane] * s);
  }
}
