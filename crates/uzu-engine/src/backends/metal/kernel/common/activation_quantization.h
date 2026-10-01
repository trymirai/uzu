#pragma once

#include <metal_stdlib>
#include "defines.h"
#include "thread_context.h"

using namespace metal;

UZU_CONST uint ACTIVATION_QUANT_TILE_SIZE = 128;
UZU_CONST float ACTIVATION_QUANT_INT8_MAX = 127.0f;
#define ACTIVATION_QUANT_SIMDGROUPS 4

static METAL_FUNC float int8_activation_scale(const float maximum_magnitude) {
  if (isfinite(maximum_magnitude) && maximum_magnitude > 0.0f) {
    return maximum_magnitude / ACTIVATION_QUANT_INT8_MAX;
  }
  return 1.0f;
}

static METAL_FUNC int8_t quantize_activation_int8(const float value, const float scale) {
  return static_cast<int8_t>(clamp(round(value / scale), -ACTIVATION_QUANT_INT8_MAX, ACTIVATION_QUANT_INT8_MAX));
}

static METAL_FUNC uint nibble_grouped_index(const uint index) {
  constexpr uint NIBBLES_PER_BYTE = 2;
  constexpr uint CODES_PER_WORD = sizeof(uint) * NIBBLES_PER_BYTE;
  constexpr uint NIBBLE_GROUP_SIZE = CODES_PER_WORD / NIBBLES_PER_BYTE;
  // [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 4, 1, 5, 2, 6, 3, 7]
  const uint offset = index % CODES_PER_WORD;
  return index - offset + (offset % NIBBLES_PER_BYTE) * NIBBLE_GROUP_SIZE + offset / NIBBLES_PER_BYTE;
}

template <typename T, typename SimdReduce, typename Combine>
METAL_FUNC T reduce_activation_quantization_group(
    const T value,
    const uint group_size,
    threadgroup T* partials,
    const thread ThreadContext& thread_context,
    SimdReduce simd_reduce,
    Combine combine
) {
  const ushort lane_index = thread_context.simd_lane_id;
  const ushort simdgroup_index = thread_context.simdgroup_index;
  const T simdgroup_value = simd_reduce(value);
  if (group_size == METAL_SIMD_SIZE) {
    return simdgroup_value;
  }
  if (lane_index == 0) {
    partials[simdgroup_index] = simdgroup_value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint simdgroups_per_group = group_size / METAL_SIMD_SIZE;
  const uint simdgroup_base = simdgroup_index / simdgroups_per_group * simdgroups_per_group;
  T result = partials[simdgroup_base];
  METAL_PRAGMA_UNROLL
  for (uint index = 1; index < simdgroups_per_group; ++index) {
    result = combine(result, partials[simdgroup_base + index]);
  }
  return result;
}

template <uint SIMDGROUPS>
METAL_FUNC float reduce_activation_quantization_row_maximum(
    const float magnitude,
    threadgroup float* partials,
    const thread ThreadContext& thread_context
) {
  static_assert(SIMDGROUPS <= METAL_SIMD_SIZE, "one lane per simdgroup partial");
  const ushort lane_index = thread_context.simd_lane_id;
  const float simdgroup_maximum = simd_max(magnitude);
  if (lane_index == 0) {
    partials[thread_context.simdgroup_index] = simdgroup_maximum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // magnitudes are non-negative, so 0 is the identity of the maximum
  return simd_max(lane_index < SIMDGROUPS ? partials[lane_index] : 0.0f);
}

template <typename T>
METAL_FUNC void write_activation_quantization_group(
    device T* output,
    const T value,
    const uint group_size,
    const uint element_count,
    const uint tile_index,
    const uint batch_index,
    const thread ThreadContext& thread_context
) {
  const ushort simdgroup_index = thread_context.simdgroup_index;
  const uint simdgroups_per_group = group_size / METAL_SIMD_SIZE;
  const uint simdgroup_offset = tile_index * ACTIVATION_QUANT_TILE_SIZE + simdgroup_index * METAL_SIMD_SIZE;
  if (thread_context.simd_lane_id == 0 && simdgroup_offset < element_count &&
      (simdgroup_index + 1) % simdgroups_per_group == 0) {
    const uint groups_per_row = element_count / group_size;
    const uint groups_per_tile = ACTIVATION_QUANT_TILE_SIZE / group_size;
    const uint group_index = tile_index * groups_per_tile + simdgroup_index / simdgroups_per_group;
    output[batch_index * groups_per_row + group_index] = value;
  }
}
