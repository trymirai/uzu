#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../matmul/common/fragment.h"
#include "../matmul/common/mxu_fragment/ops.h"

using namespace metal;

using Ops = uzu::matmul::MxuFragmentOps<>;

// Codebook levels of a trellis state. Byte j of fmix32(state * 0xCFCCB83F + 0x584B4AA3) gives level
// 8 * (sum of its four 2-bit fields) + (3 * low nibble mod 16) - 54 in [-54, 57], returned as signed bytes.
// The weight of component j is scale * level + offset[j] (the loader checks the package codebook against this).
static METAL_FUNC uint trellis_levels(uint state) {
  uint x = state * 0xCFCCB83Fu + 0x584B4AA3u;
  x ^= x >> 16;
  x *= 0x85EBCA6Bu;
  x ^= x >> 16;
  // each nibble is a + 4b with 2-bit fields a, b: minus 3b leaves a + b in [0, 6], so no borrow crosses nibbles
  const uint nibble_sums = x - 3 * ((x >> 2) & 0x33333333u);
  const uint field_sums = (nibble_sums + (nibble_sums >> 4)) & 0x0F0F0F0Fu;
  const uint plus_54 = (field_sums << 3) + (((x & 0x0F0F0F0Fu) * 3) & 0x0F0F0F0Fu);
  return (plus_54 + 0x4A4A4A4Au) ^ 0x80808080u;
}

// V4 rows restart every 64 columns: 17-byte blocks of a 16-bit little-endian seed state and one 8-bit
// transition per later group, so group g >= 2 reads bytes (g, g + 1).
static METAL_FUNC uint restart_state(device const uchar* block, uint group) {
  const uint a = block[group];
  const uint b = block[group + 1];
  const uint high = group == 1 ? uint(block[0]) : (group == 0 ? b : a);
  const uint low = group == 0 ? a : b;
  return (high << 8) | low;
}

// States of groups (group, group + 1) of a V2 row the loader repacked MSB-first (16-bit seed, then
// TRANSITION_BITS per group), so a state is the 16-bit window at bit TRANSITION_BITS * group.
template <uint TRANSITION_BITS>
static METAL_FUNC ushort2 state_pair(device const uchar* row, uint group) {
  const uint bit = group * TRANSITION_BITS;
  device const uchar* bytes = row + bit / 8;
  const uint window = (uint(bytes[0]) << 24) | (uint(bytes[1]) << 16) | (uint(bytes[2]) << 8) |
                      (TRANSITION_BITS == 4 ? 0 : uint(bytes[3]));
  const uint shift = bit % 8;
  return ushort2(window >> (16 - shift), window >> (16 - TRANSITION_BITS - shift));
}

// f32 [scale, offset of column class 0..4, 0, 0, 0], then for V2 the int8 level pair of every state.
struct Codebook {
  device const float* values;

  METAL_FUNC float scale() const { return values[0]; }
  METAL_FUNC float4 offsets() const { return float4(values[1], values[2], values[3], values[4]); }
  METAL_FUNC char2 pair_levels(uint state) const { return reinterpret_cast<device const char2*>(values + 8)[state]; }
};

// Levels of columns column..column + 3 of one row. A V2 state carries two levels, so the hash costs twice as
// much per weight as for V4; READ_V2_LEVELS reads them from the codebook's level table instead.
template <uint VECTOR_WIDTH, uint TRANSITION_BITS, bool READ_V2_LEVELS>
static METAL_FUNC char4 decode_columns(device const uchar* row, uint column, Codebook codebook) {
  if (VECTOR_WIDTH == 4) {
    return as_type<char4>(trellis_levels(restart_state(row + column / 64 * 17, column % 64 / 4)));
  }
  const ushort2 states = state_pair<TRANSITION_BITS>(row, column / 2);
  if (READ_V2_LEVELS) {
    return char4(codebook.pair_levels(states.x), codebook.pair_levels(states.y));
  }
  return char4(as_type<char2>(ushort(trellis_levels(states.x))), as_type<char2>(ushort(trellis_levels(states.y))));
}

// Levels of 16 rows x 32 columns into MXU operand fragments (fragment_row, 0) and (fragment_row, 1): a lane
// holds rows (y, y + 8) x columns x..x + 3 of each 16-column half, (x, y) being its fragment position.
template <uint VECTOR_WIDTH, uint TRANSITION_BITS, bool READ_V2_LEVELS, typename Fragments>
static METAL_FUNC void decode_rows(
    thread Fragments& fragments,
    ushort fragment_row,
    device const uchar* upper_codes,
    device const uchar* lower_codes,
    uint chunk,
    short2 position,
    Codebook codebook
) {
  METAL_PRAGMA_UNROLL
  for (ushort column_half = 0; column_half < 2; ++column_half) {
    const uint column = chunk * 32 + column_half * 16 + position.x;
    const char4 upper = decode_columns<VECTOR_WIDTH, TRANSITION_BITS, READ_V2_LEVELS>(upper_codes, column, codebook);
    const char4 lower = decode_columns<VECTOR_WIDTH, TRANSITION_BITS, READ_V2_LEVELS>(lower_codes, column, codebook);
    thread auto& values = fragments.fragment_at(fragment_row, column_half);
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < 4; ++i) {
      values[i] = upper[i];
      values[4 + i] = lower[i];
    }
  }
}

// output[token][row] = (scale * sum_k level[row][k] * a[token][k] + offsets . class_sums[token])
//                      * row_scales[row] * activation_scale[token]
// for weights scale * level + offsets[k % 4] and the int8 activations a and token statistics
// (class_sums, (activation_scale, 0, 0, 0)) of MiraiSTransform.
static METAL_FUNC void store_output(
    int32_t level_dot,
    uint token,
    uint row,
    Codebook codebook,
    device const float4* token_statistics,
    device const float* row_scales,
    device bfloat* output,
    uint output_stride
) {
  const float4 class_sums = token_statistics[2 * token];
  const float activation_scale = token_statistics[2 * token + 1].x;
  const float dot = float(level_dot) * codebook.scale() + metal::dot(class_sums, codebook.offsets());
  output[token * output_stride + row] = bfloat(dot * row_scales[row] * activation_scale);
}

static METAL_FUNC uint row_bytes(uint vector_width, uint transition_bits, uint columns) {
  return vector_width == 4 ? columns / 64 * 17 : (16 + (columns / 2 - 1) * transition_bits + 7) / 8;
}

#define WIDE_SIMDGROUPS 4

// Rows as the MXU M dimension: each SIMDgroup decodes 16 rows once per TOKENS tokens and the MXU reads the
// activations straight from device memory. `rows` is a multiple of 16.
template <uint TOKENS, uint VECTOR_WIDTH, uint TRANSITION_BITS>
VARIANTS(TOKENS, 32, 64)
VARIANTS(VECTOR_WIDTH, 2, 4)
VARIANTS(TRANSITION_BITS, 4, 6, 8)
CONSTRAINT((VECTOR_WIDTH == 4) == (TRANSITION_BITS == 8))
KERNEL(MiraiSProjection)(
    device const uchar* codes,
    device const int8_t* activations,
    device const float4* token_statistics,
    device const float* row_scales,
    device const float* codebook,
    device bfloat* output,
    constant uint& rows,
    constant uint& columns,
    constant uint& batch,
    constant uint& output_stride,
    const uint row_tile GROUPS(rows.div_ceil(WIDE_SIMDGROUPS * 16)),
    const uint token_tile GROUPS(batch.div_ceil(TOKENS)),
    const uint thread_index THREADS(WIDE_SIMDGROUPS * 32),
    const ThreadContext thread_context
) {
  (void)thread_index;
  using WeightTile = uzu::matmul::Fragment<int8_t, 1, 2, Ops>;
  using Accumulator = uzu::matmul::Fragment<int32_t, 1, TOKENS / 16, Ops>;
  const uint row_base = (row_tile * WIDE_SIMDGROUPS + thread_context.simdgroup_index) * 16;
  if (row_base >= rows) {
    return;
  }
  const Codebook trellis_codebook{codebook};
  const short2 position = Ops::get_position(thread_context.simd_lane_id);
  const uint stride = row_bytes(VECTOR_WIDTH, TRANSITION_BITS, columns);
  device const uchar* upper_codes = codes + (row_base + position.y) * stride;
  device const int8_t* tile_activations = activations + token_tile * TOKENS * columns;

  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      16,
      32,
      32,
      false,
      true,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
  );
  mpp::tensor_ops::matmul2d<descriptor, execution_simdgroup> matmul;
  Accumulator accumulator;
  accumulator.clear();
  for (uint chunk = 0; chunk < columns / 32; ++chunk) {
    WeightTile weights;
    decode_rows<VECTOR_WIDTH, TRANSITION_BITS, false>(
        weights,
        0,
        upper_codes,
        upper_codes + 8 * stride,
        chunk,
        position,
        trellis_codebook
    );
    auto left = matmul.template get_left_input_cooperative_tensor<int8_t, int8_t, int>();
    Ops::load_paired_vectors(left, weights.fragment_at(0, 0), weights.fragment_at(0, 1));
    METAL_PRAGMA_UNROLL
    for (ushort token_fragment = 0; token_fragment < TOKENS / 16; token_fragment += 2) {
      // activations [32 tokens x 32 columns], row stride `columns`
      tensor<device int8_t, extents<int, 32, 32>, tensor_inline> right(
          const_cast<device int8_t*>(tile_activations + token_fragment * 16 * columns + chunk * 32),
          extents<int, 32, 32>{},
          array<int, 2>{1, int(columns)}
      );
      auto destination = matmul.template get_destination_cooperative_tensor<decltype(left), decltype(right), int>();
      thread auto& first = accumulator.fragment_at(0, token_fragment);
      thread auto& second = accumulator.fragment_at(0, token_fragment + 1);
      Ops::load_paired_vectors(destination, first, second);
      matmul.run(left, right, destination);
      Ops::store_paired_vectors(destination, first, second);
    }
  }

  accumulator.map_coords(thread_context.simd_lane_id, [&](short row, short token_index, int32_t value) {
    const uint token = token_tile * TOKENS + token_index;
    if (token < batch) {
      store_output(value, token, row_base + row, trellis_codebook, token_statistics, row_scales, output, output_stride);
    }
    return value;
  });
}

#define NARROW_SIMDGROUPS 2

// Tokens as the MXU M dimension (16 per tile) for small batches: each SIMDgroup decodes ROW_FRAGMENTS x 16 rows
// against one register tile of activations, so the MXU does no padded token work.
template <uint ROW_FRAGMENTS, uint VECTOR_WIDTH, uint TRANSITION_BITS>
VARIANTS(ROW_FRAGMENTS, 2, 4)
VARIANTS(VECTOR_WIDTH, 2, 4)
VARIANTS(TRANSITION_BITS, 4, 6, 8)
CONSTRAINT((VECTOR_WIDTH == 4) == (TRANSITION_BITS == 8))
KERNEL(MiraiSNarrowProjection)(
    device const uchar* codes,
    device const int8_t* activations,
    device const float4* token_statistics,
    device const float* row_scales,
    device const float* codebook,
    device bfloat* output,
    constant uint& rows,
    constant uint& columns,
    constant uint& batch,
    constant uint& output_stride,
    const uint row_tile GROUPS(rows.div_ceil(NARROW_SIMDGROUPS * ROW_FRAGMENTS * 16)),
    const uint token_tile GROUPS(batch.div_ceil(16)),
    const uint thread_index THREADS(NARROW_SIMDGROUPS * 32),
    const ThreadContext thread_context
) {
  (void)thread_index;
  using ActivationTile = uzu::matmul::Fragment<int8_t, 1, 2, Ops>;
  using WeightTiles = uzu::matmul::OperandFragment<int8_t, 2, ROW_FRAGMENTS, Ops, uzu::matmul::ReadTranspose>;
  using Accumulator = uzu::matmul::Fragment<int32_t, 1, ROW_FRAGMENTS, Ops>;
  const uint row_base = (row_tile * NARROW_SIMDGROUPS + thread_context.simdgroup_index) * ROW_FRAGMENTS * 16;
  if (row_base >= rows) {
    return;
  }
  const Codebook trellis_codebook{codebook};
  const short2 position = Ops::get_position(thread_context.simd_lane_id);
  const uint stride = row_bytes(VECTOR_WIDTH, TRANSITION_BITS, columns);
  device const int8_t* tile_activations = activations + token_tile * 16 * columns;

  Accumulator accumulator;
  accumulator.clear();
  for (uint chunk = 0; chunk < columns / 32; ++chunk) {
    WeightTiles weights;
    METAL_PRAGMA_UNROLL
    for (ushort fragment_row = 0; fragment_row < ROW_FRAGMENTS; ++fragment_row) {
      // rows past `rows` decode the last row and are never stored
      const uint row = row_base + fragment_row * 16 + position.y;
      decode_rows<VECTOR_WIDTH, TRANSITION_BITS, true>(
          weights,
          fragment_row,
          codes + min(row, rows - 1) * stride,
          codes + min(row + 8, rows - 1) * stride,
          chunk,
          position,
          trellis_codebook
      );
    }
    ActivationTile activation_tile;
    activation_tile.load_from(
        thread_context.simd_lane_id,
        uzu::matmul::fragment_source(tile_activations + chunk * 32, int(columns))
    );
    uzu::matmul::fragment_mma(accumulator, activation_tile, weights);
  }

  accumulator.map_coords(thread_context.simd_lane_id, [&](short token_index, short row, int32_t value) {
    const uint token = token_tile * 16 + token_index;
    if (token < batch && row_base + row < rows) {
      store_output(value, token, row_base + row, trellis_codebook, token_statistics, row_scales, output, output_stride);
    }
    return value;
  });
}

#define SIMDGROUP_KERNEL_SIMDGROUPS 4
#define SIMDGROUP_KERNEL_ROWS 4

static METAL_FUNC int level_dot(char4 levels, char4 activations) {
  const int4 products = int4(levels) * int4(activations);
  return products.x + products.y + products.z + products.w;
}

// Without MXU (Apple GPUs before M5), plain SIMD arithmetic: each SIMDgroup computes SIMDGROUP_KERNEL_ROWS rows x
// TOKENS tokens, a lane decodes 4 columns of every row per 128-column step, and simd_sum adds up the lanes. The int32
// dots are exact, so the output matches the MXU kernels bit for bit. V2 levels are hashed too: on M1 and M2 reading
// them from the level table is 2.5-3x slower at batch 1. `rows` is a multiple of 16.
template <uint TOKENS, uint VECTOR_WIDTH, uint TRANSITION_BITS>
VARIANTS(TOKENS, 1, 8)
VARIANTS(VECTOR_WIDTH, 2, 4)
VARIANTS(TRANSITION_BITS, 4, 6, 8)
CONSTRAINT((VECTOR_WIDTH == 4) == (TRANSITION_BITS == 8))
KERNEL(MiraiSSimdgroupProjection)(
    device const uchar* codes,
    device const int8_t* activations,
    device const float4* token_statistics,
    device const float* row_scales,
    device const float* codebook,
    device bfloat* output,
    constant uint& rows,
    constant uint& columns,
    constant uint& batch,
    constant uint& output_stride,
    const uint row_tile GROUPS(rows.div_ceil(SIMDGROUP_KERNEL_SIMDGROUPS * SIMDGROUP_KERNEL_ROWS)),
    const uint token_tile GROUPS(batch.div_ceil(TOKENS)),
    const uint thread_index THREADS(SIMDGROUP_KERNEL_SIMDGROUPS * 32),
    const ThreadContext thread_context
) {
  (void)thread_index;
  static_assert(SIMDGROUP_KERNEL_ROWS * TOKENS <= 32, "each lane stores one output");
  const uint row_base =
      (row_tile * SIMDGROUP_KERNEL_SIMDGROUPS + thread_context.simdgroup_index) * SIMDGROUP_KERNEL_ROWS;
  const Codebook trellis_codebook{codebook};
  const uint stride = row_bytes(VECTOR_WIDTH, TRANSITION_BITS, columns);
  device const int8_t* tile_activations = activations + token_tile * TOKENS * columns;

  int dots[SIMDGROUP_KERNEL_ROWS][TOKENS] = {};
  for (uint column = thread_context.simd_lane_id * 4; column < columns; column += 128) {
    char4 token_activations[TOKENS];
    METAL_PRAGMA_UNROLL
    for (uint token = 0; token < TOKENS; ++token) {
      token_activations[token] = *reinterpret_cast<device const char4*>(tile_activations + token * columns + column);
    }
    METAL_PRAGMA_UNROLL
    for (uint row = 0; row < SIMDGROUP_KERNEL_ROWS; ++row) {
      device const uchar* row_codes = codes + (row_base + row) * stride;
      const char4 levels = decode_columns<VECTOR_WIDTH, TRANSITION_BITS, false>(row_codes, column, trellis_codebook);
      METAL_PRAGMA_UNROLL
      for (uint token = 0; token < TOKENS; ++token) {
        dots[row][token] += level_dot(levels, token_activations[token]);
      }
    }
  }

  METAL_PRAGMA_UNROLL
  for (uint row = 0; row < SIMDGROUP_KERNEL_ROWS; ++row) {
    METAL_PRAGMA_UNROLL
    for (uint token = 0; token < TOKENS; ++token) {
      const int dot = simd_sum(dots[row][token]);
      const uint output_token = token_tile * TOKENS + token;
      if (thread_context.simd_lane_id == row * TOKENS + token && output_token < batch) {
        store_output(
            dot,
            output_token,
            row_base + row,
            trellis_codebook,
            token_statistics,
            row_scales,
            output,
            output_stride
        );
      }
    }
  }
}
