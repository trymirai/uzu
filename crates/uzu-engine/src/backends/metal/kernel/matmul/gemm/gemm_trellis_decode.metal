#include "../../common/dsl.h"
#include "../common/quant_unpack.h"
#include "common/quantized/trellis.h"

using namespace metal;
using namespace uzu::gemm;

#define SIMDGROUPS 8
#define ROWS_PER_SIMDGROUP 4

UZU_CONST uint ROWS_PER_THREADGROUP = ROWS_PER_SIMDGROUP * SIMDGROUPS;
UZU_CONST uint MAX_BATCH = TRELLIS_DECODE_MAX_BATCH;

// Pre-M5 batch 1..8. Lane partials are exact integers in float, so the cross-lane sum is an int simd_sum.
KERNEL(GemmTrellisDecode)(
    const device uint8_t* activations,
    const device float4* column_group_sums,
    const device float* activation_scales,
    const device uint8_t* codes,
    const device float* row_scales,
    const device float* scale_and_offsets,
    device bfloat* output,
    const constant uint& k,
    const constant uint& n,
    const constant uint& output_stride,
    const uint batch SPECIALIZE,
    const uint vector_width SPECIALIZE,
    const uint transition_bits SPECIALIZE,
    const uint restart_columns SPECIALIZE,
    const uint row_block GROUPS(n.div_ceil(ROWS_PER_SIMDGROUP* SIMDGROUPS)),
    const uint lane THREADS(METAL_SIMD_SIZE),
    const uint simdgroup THREADS(SIMDGROUPS)
) {
  const trellis::GemmTrellisFormat format{vector_width, transition_bits, restart_columns};
  const uint row_bytes = trellis::row_bytes(format, k);
  const uint first_row = row_block * ROWS_PER_THREADGROUP + simdgroup * ROWS_PER_SIMDGROUP;
  // Rows past n re-read the last row.
  uint row_offsets[ROWS_PER_SIMDGROUP];
  if (batch <= 2) {
    METAL_PRAGMA_UNROLL
    for (uint row = 0; row < ROWS_PER_SIMDGROUP; ++row) {
      row_offsets[row] = min(first_row + row, n - 1) * row_bytes;
    }
  }
  float sums[MAX_BATCH][ROWS_PER_SIMDGROUP] = {};

  for (uint column = lane * trellis::LANE_COLUMNS; column < k; column += METAL_SIMD_SIZE * trellis::LANE_COLUMNS) {
    uint4 words[ROWS_PER_SIMDGROUP];
    METAL_PRAGMA_UNROLL
    for (uint row = 0; row < ROWS_PER_SIMDGROUP; ++row) {
      words[row] = trellis::decode_biased_weight_quad(
          format,
          codes,
          batch <= 2 ? row_offsets[row] : min(first_row + row, n - 1) * row_bytes,
          column,
          k
      );
    }
    uint4 inputs[MAX_BATCH];
    METAL_PRAGMA_UNROLL
    for (uint token = 0; token < MAX_BATCH; ++token) {
      if (token < batch) {
        inputs[token] = *reinterpret_cast<const device uint4*>(activations + size_t(token) * k + column);
      }
    }
    METAL_PRAGMA_UNROLL
    for (uint quad = 0; quad < 4; ++quad) {
      float4 weights[ROWS_PER_SIMDGROUP];
      METAL_PRAGMA_UNROLL
      for (uint row = 0; row < ROWS_PER_SIMDGROUP; ++row) {
        weights[row] = bytes_to_float(words[row][quad], float(trellis::WEIGHT_BIAS));
      }
      METAL_PRAGMA_UNROLL
      for (uint token = 0; token < MAX_BATCH; ++token) {
        if (token < batch) {
          const float4 input = int8_to_float(inputs[token][quad]);
          METAL_PRAGMA_UNROLL
          for (uint row = 0; row < ROWS_PER_SIMDGROUP; ++row) {
            METAL_PRAGMA_UNROLL
            for (uint lane_column = 0; lane_column < 4; ++lane_column) {
              sums[token][row] = fma(weights[row][lane_column], input[lane_column], sums[token][row]);
            }
          }
        }
      }
    }
  }

  const float4 offsets = float4(scale_and_offsets[1], scale_and_offsets[2], scale_and_offsets[3], scale_and_offsets[4]);
  METAL_PRAGMA_UNROLL
  for (uint token = 0; token < MAX_BATCH; ++token) {
    if (token < batch) {
      const float offsets_dot = dot(column_group_sums[token], offsets);
      METAL_PRAGMA_UNROLL
      for (uint row = 0; row < ROWS_PER_SIMDGROUP; ++row) {
        const float level_dot = float(simd_sum(int(sums[token][row])));
        if (lane == 0 && first_row + row < n) {
          output[token * output_stride + first_row + row] = bfloat(
              (level_dot * scale_and_offsets[0] + offsets_dot) * row_scales[first_row + row] * activation_scales[token]
          );
        }
      }
    }
  }
}
