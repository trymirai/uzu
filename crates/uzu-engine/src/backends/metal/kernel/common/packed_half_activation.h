#pragma once

#include <metal_stdlib>

#include "activation_quantization.h"
#include "../matmul/gemm/common/packed_half_layout.h"

using namespace metal;

UZU_CONST uint PACKED_HALF_ROWS_PER_WORD = 2;
UZU_CONST uint PACKED_HALF_VALUES_PER_VECTOR = 4;
UZU_CONST float PACKED_HALF_MAX_FINITE = 65504.0f;

// Matches the exponent offsets embedded in w4_pair's four half-precision nibble encodings.
UZU_CONST float W4_PAIR_BIAS_0 = 1024.0f;
UZU_CONST float W4_PAIR_BIAS_1 = 64.0f;
UZU_CONST float W4_PAIR_BIAS_2 = 256.0f;
UZU_CONST float W4_PAIR_BIAS_3 = 16.0f;

static METAL_FUNC float w4_pair_bias(uint component) {
  return component == 0   ? W4_PAIR_BIAS_0
         : component == 1 ? W4_PAIR_BIAS_1
         : component == 2 ? W4_PAIR_BIAS_2
                          : W4_PAIR_BIAS_3;
}

static METAL_FUNC float2 store_packed_half_values(
    const float4 values,
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
  for (uint i = 0; i < PACKED_HALF_VALUES_PER_VECTOR; ++i) {
    const half value = half(clamp(values[i], -PACKED_HALF_MAX_FINITE, PACKED_HALF_MAX_FINITE));
    halves[PACKED_HALF_ROWS_PER_WORD * (word + i) + row % PACKED_HALF_ROWS_PER_WORD] = value;
    sums += float2(float(value), w4_pair_bias(i) * float(value));
  }
  return sums;
}

static METAL_FUNC void store_packed_half_sums(
    float2 sums,
    const uint lane,
    const uint lane_count,
    const uint group,
    const uint row,
    const uint group_size,
    device uint* packed
) {
  using Layout32 = uzu::gemm::PackedHalfLayout<32>;
  using Layout64 = uzu::gemm::PackedHalfLayout<64>;
  sums.x = reduce_adjacent_lanes(sums.x, lane_count, [](float x, float y) { return x + y; });
  sums.y = reduce_adjacent_lanes(sums.y, lane_count, [](float x, float y) { return x + y; });
  if (lane % lane_count == 0) {
    const uint group_words = group_size == 32 ? Layout32::GROUP_WORDS : Layout64::GROUP_WORDS;
    const uint sums_offset = group_size == 32 ? Layout32::sums_offset() : Layout64::sums_offset();
    device float* output = reinterpret_cast<device float*>(packed + group * group_words + sums_offset);
    output[row * 2] = sums.x;
    output[row * 2 + 1] = sums.y;
  }
}
