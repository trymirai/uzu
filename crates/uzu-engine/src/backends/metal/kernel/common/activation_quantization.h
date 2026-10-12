#pragma once

#include <metal_stdlib>
#include "defines.h"
#include "thread_context.h"

using namespace metal;

UZU_CONST float ACTIVATION_QUANT_INT8_MAX = 127.0f;

static METAL_FUNC int8_t quantize_activation_int8(const float value, const float scale) {
  return static_cast<int8_t>(clamp(round(value / scale), -ACTIVATION_QUANT_INT8_MAX, ACTIVATION_QUANT_INT8_MAX));
}

template <uint SIMDGROUPS>
METAL_FUNC float reduce_activation_quantization_row_maximum(
    const float largest_abs_value,
    threadgroup float* partials,
    const thread ThreadContext& thread_context
) {
  static_assert(SIMDGROUPS <= METAL_SIMD_SIZE, "one lane per simdgroup partial");
  const ushort lane_index = thread_context.simd_lane_id;
  const float simdgroup_maximum = simd_max(largest_abs_value);
  if (lane_index == 0) {
    partials[thread_context.simdgroup_index] = simdgroup_maximum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // magnitudes are non-negative, so 0 is the identity of the group_max_abs
  return simd_max(lane_index < SIMDGROUPS ? partials[lane_index] : 0.0f);
}

UZU_CONST uint ACTIVATION_QUANT_VECTOR_SIZE = 4;

template <typename T, typename Combine>
static METAL_FUNC T reduce_adjacent_lanes(T value, const uint lane_count, Combine combine) {
  for (ushort lane_stride = 1; lane_stride < lane_count; lane_stride <<= 1) {
    value = combine(value, simd_shuffle_xor(value, lane_stride));
  }
  return value;
}

static METAL_FUNC void store_quantized_activation_vector(
    const float4 values,
    const ushort lane_index,
    const bool in_bounds,
    const uint first_index,
    const uint row_width,
    const uint batch_index,
    const uint scale_group_size,
    const bool grouped_by_weight_nibble,
    const bool emit_group_sums,
    const uint sum_group_size,
    device int8_t* q_out,
    device float* scales_out,
    device int32_t* group_sums_out
) {
  const float largest_abs_value = max(max(fabs(values[0]), fabs(values[1])), max(fabs(values[2]), fabs(values[3])));
  const uint lanes_per_scale_group = scale_group_size / ACTIVATION_QUANT_VECTOR_SIZE;
  const float group_max_abs =
      reduce_adjacent_lanes(largest_abs_value, lanes_per_scale_group, [](float x, float y) { return max(x, y); });
  const float scale =
      isfinite(group_max_abs) && group_max_abs > 0.0f ? group_max_abs / ACTIVATION_QUANT_INT8_MAX : 1.0f;
  int8_t codes[ACTIVATION_QUANT_VECTOR_SIZE];
  int thread_code_sum = 0;
  uint packed_codes = 0;
  METAL_PRAGMA_UNROLL
  for (uint index = 0; index < ACTIVATION_QUANT_VECTOR_SIZE; ++index) {
    codes[index] = quantize_activation_int8(values[index], scale);
    thread_code_sum += static_cast<int>(codes[index]);
    packed_codes |= static_cast<uint>(static_cast<uchar>(codes[index])) << (8 * index);
  }

  const uint row_start = batch_index * row_width;
  if (grouped_by_weight_nibble) {
    // Word order [0..7] -> [0, 4, 1, 5, 2, 6, 3, 7]: the even lane writes codes (0, 2, 4, 6), the odd lane (1, 3, 5,
    // 7).
    const uint neighbor_codes = simd_shuffle_xor(packed_codes, 1);
    const uint is_odd_lane = lane_index & 1;
    const uint even_codes = is_odd_lane ? neighbor_codes : packed_codes;
    const uint odd_codes = is_odd_lane ? packed_codes : neighbor_codes;
    if (in_bounds) {
      const uint shift = 8 * is_odd_lane;
      const packed_char4 word = packed_char4(
          static_cast<char>(even_codes >> shift),
          static_cast<char>(even_codes >> (shift + 16)),
          static_cast<char>(odd_codes >> shift),
          static_cast<char>(odd_codes >> (shift + 16))
      );
      *reinterpret_cast<device packed_char4*>(q_out + row_start + (first_index & ~7u) + 4 * is_odd_lane) = word;
    }
  } else if (in_bounds) {
    *reinterpret_cast<device packed_char4*>(q_out + row_start + first_index) =
        packed_char4(codes[0], codes[1], codes[2], codes[3]);
  }

  if (in_bounds && lane_index % lanes_per_scale_group == 0) {
    scales_out[batch_index * (row_width / scale_group_size) + first_index / scale_group_size] = scale;
  }

  if (emit_group_sums) {
    const uint lanes_per_sum_group = sum_group_size / ACTIVATION_QUANT_VECTOR_SIZE;
    const int sum = reduce_adjacent_lanes(thread_code_sum, lanes_per_sum_group, [](int x, int y) { return x + y; });
    if (in_bounds && lane_index % lanes_per_sum_group == 0) {
      group_sums_out[batch_index * (row_width / sum_group_size) + first_index / sum_group_size] = sum;
    }
  }
}
