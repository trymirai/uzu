#pragma once

#include <metal_stdlib>

#include "activation_quantization.h"
#include "../matmul/common/packed_half_layout.h"
#include "../matmul/common/quant_unpack.h"

using namespace metal;

UZU_CONST float PACKED_HALF_MAX_FINITE = 65504.0f;

static METAL_FUNC void store_packed_half(
    const float4 values,
    const uint lane,
    const uint first_index,
    const uint row,
    const uint block_k,
    const uint weight_bits,
    device uint* packed
) {
  using Layout32 = uzu::gemm::PackedHalfLayout<32, 4>;
  using Layout64 = uzu::gemm::PackedHalfLayout<64, 4>;
  using Layout128 = uzu::gemm::PackedHalfLayout<128, 2>;
  const uint group = first_index / block_k;
  const uint k = first_index % block_k;
  const uint sums_offset = METAL_SIMD_SIZE * (block_k / Layout32::MMA_WIDTH);
  const uint group_words =
      sums_offset + (METAL_SIMD_SIZE / Layout32::MMA_WIDTH) * Layout32::ROWS_PER_PAIR * Layout32::SUMS_PER_ROW;
  const uint word = group * group_words + (weight_bits == 2 ? Layout128::word_offset(k, row)
                                           : block_k == 32  ? Layout32::word_offset(k, row)
                                                            : Layout64::word_offset(k, row));
  device half* halves = reinterpret_cast<device half*>(packed);
  float2 sums = 0.0f;
  for (uint i = 0; i < Layout32::VALUES_PER_VECTOR; ++i) {
    const half value = half(clamp(values[i], -PACKED_HALF_MAX_FINITE, PACKED_HALF_MAX_FINITE));
    halves[Layout32::ROWS_PER_PAIR * (word + i) + row % Layout32::ROWS_PER_PAIR] = value;
    const float bias = weight_bits == 2 ? uzu::gemm::w2_pair_bias_value((k + i) % 8) : uzu::gemm::w4_pair_bias_value(i);
    sums += float2(float(value), bias * float(value));
  }
  const uint lane_count = block_k / Layout32::VALUES_PER_VECTOR;
  sums.x = reduce_adjacent_lanes(sums.x, lane_count, [](float x, float y) { return x + y; });
  sums.y = reduce_adjacent_lanes(sums.y, lane_count, [](float x, float y) { return x + y; });
  if (lane % lane_count == 0) {
    device float* output = reinterpret_cast<device float*>(packed + group * group_words + sums_offset);
    output[row * Layout32::SUMS_PER_ROW] = sums.x;
    output[row * Layout32::SUMS_PER_ROW + 1] = sums.y;
  }
}
