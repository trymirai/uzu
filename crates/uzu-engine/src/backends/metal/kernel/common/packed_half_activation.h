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
    const uint group_size,
    device uint* packed
) {
  using Layout32 = uzu::gemm::PackedHalfLayout<32>;
  using Layout64 = uzu::gemm::PackedHalfLayout<64>;
  const uint group = first_index / group_size;
  const uint k = first_index % group_size;
  const uint group_words = group_size == 32 ? Layout32::GROUP_WORDS : Layout64::GROUP_WORDS;
  const uint word =
      group * group_words + (group_size == 32 ? Layout32::word_offset(k, row) : Layout64::word_offset(k, row));
  device half* halves = reinterpret_cast<device half*>(packed);
  float2 sums = 0.0f;
  for (uint i = 0; i < Layout32::VALUES_PER_VECTOR; ++i) {
    const half value = half(clamp(values[i], -PACKED_HALF_MAX_FINITE, PACKED_HALF_MAX_FINITE));
    halves[Layout32::ROWS_PER_PAIR * (word + i) + row % Layout32::ROWS_PER_PAIR] = value;
    sums += float2(float(value), uzu::gemm::w4_pair_bias_value(i) * float(value));
  }
  const uint lane_count = group_size / Layout32::VALUES_PER_VECTOR;
  sums.x = reduce_adjacent_lanes(sums.x, lane_count, [](float x, float y) { return x + y; });
  sums.y = reduce_adjacent_lanes(sums.y, lane_count, [](float x, float y) { return x + y; });
  if (lane % lane_count == 0) {
    const uint sums_offset = group_size == 32 ? Layout32::sums_offset() : Layout64::sums_offset();
    device float* output = reinterpret_cast<device float*>(packed + group * group_words + sums_offset);
    output[row * Layout32::SUMS_PER_ROW] = sums.x;
    output[row * Layout32::SUMS_PER_ROW + 1] = sums.y;
  }
}
