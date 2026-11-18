#include "../../common/dsl.h"
#include "../../generated/trellis.h"
#include "../common/quant_unpack.h"
#include "common/quantized/trellis.h"

using namespace metal;
using namespace uzu::gemm;
using uzu::trellis::TrellisCodebook;

#define SIMDGROUPS 8
#define ELEMENTS_PER_SIMDGROUP 4

UZU_CONST uint ELEMENTS_PER_THREADGROUP = ELEMENTS_PER_SIMDGROUP * SIMDGROUPS;

// Output element (token, column) reads weight row gather_indices[token * n + column] - gather_first_row and is left
// untouched when that row is outside the gathered window. Lane partials are exact integers in float, as in the decode
// kernel.
KERNEL(GemmTrellisGather)(
    const device uint8_t* activations,
    const device float4* column_group_sums,
    const device float* activation_scales,
    const device uint8_t* codes,
    const device float* row_scales,
    const constant TrellisCodebook& codebook,
    const device uint* gather_indices,
    device bfloat* output,
    const constant uint& m,
    const constant uint& n,
    const constant uint& k,
    const constant uint& output_stride,
    const constant uint& gather_first_row,
    const constant uint& gather_row_count,
    const uint vector_width SPECIALIZE,
    const uint transition_bits SPECIALIZE,
    const uint restart_columns SPECIALIZE,
    const uint element_block GROUPS((m * n).div_ceil(ELEMENTS_PER_SIMDGROUP* SIMDGROUPS)),
    const uint lane THREADS(METAL_SIMD_SIZE),
    const uint simdgroup THREADS(SIMDGROUPS)
) {
  const trellis::GemmTrellisFormat format{vector_width, transition_bits, restart_columns};
  const uint row_bytes = trellis::row_bytes(format, k);
  const uint first_element = element_block * ELEMENTS_PER_THREADGROUP + simdgroup * ELEMENTS_PER_SIMDGROUP;
  // Elements past m * n and rows outside the window re-read a valid row and are not written.
  uint tokens[ELEMENTS_PER_SIMDGROUP], columns[ELEMENTS_PER_SIMDGROUP], rows[ELEMENTS_PER_SIMDGROUP];
  bool written[ELEMENTS_PER_SIMDGROUP];
  METAL_PRAGMA_UNROLL
  for (uint element = 0; element < ELEMENTS_PER_SIMDGROUP; ++element) {
    const uint index = min(first_element + element, m * n - 1);
    const uint row = gather_indices[index] - gather_first_row;
    tokens[element] = index / n;
    columns[element] = index % n;
    written[element] = first_element + element < m * n && row < gather_row_count;
    rows[element] = written[element] ? row : 0;
  }
  float sums[ELEMENTS_PER_SIMDGROUP] = {};

  for (uint column = lane * trellis::LANE_COLUMNS; column < k; column += METAL_SIMD_SIZE * trellis::LANE_COLUMNS) {
    uint4 words[ELEMENTS_PER_SIMDGROUP], inputs[ELEMENTS_PER_SIMDGROUP];
    METAL_PRAGMA_UNROLL
    for (uint element = 0; element < ELEMENTS_PER_SIMDGROUP; ++element) {
      words[element] = trellis::decode_biased_weight_quad(format, codes, rows[element] * row_bytes, column, k);
      inputs[element] = *reinterpret_cast<const device uint4*>(activations + size_t(tokens[element]) * k + column);
    }
    METAL_PRAGMA_UNROLL
    for (uint quad = 0; quad < 4; ++quad) {
      METAL_PRAGMA_UNROLL
      for (uint element = 0; element < ELEMENTS_PER_SIMDGROUP; ++element) {
        const float4 weights = bytes_to_float(words[element][quad], float(trellis::WEIGHT_BIAS));
        sums[element] += dot(weights, int8_to_float(inputs[element][quad]));
      }
    }
  }

  const float4 offsets = float4(codebook.offsets[0], codebook.offsets[1], codebook.offsets[2], codebook.offsets[3]);
  METAL_PRAGMA_UNROLL
  for (uint element = 0; element < ELEMENTS_PER_SIMDGROUP; ++element) {
    const float level_dot = float(simd_sum(int(sums[element])));
    if (lane == 0 && written[element]) {
      const uint token = tokens[element];
      const float offsets_dot = dot(column_group_sums[token], offsets);
      output[token * output_stride + columns[element]] =
          bfloat((level_dot * codebook.scale + offsets_dot) * row_scales[rows[element]] * activation_scales[token]);
    }
  }
}
