#pragma once

#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/thread_context.h"
#include "../generated/hadamard_order.h"

using namespace metal;
using uzu::hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE;

static_assert(
    HADAMARD_TRANSFORM_BLOCK_SIZE == METAL_SIMD_SIZE,
    "Hadamard transform block size must match the Metal SIMD width"
);

static METAL_FUNC float simdgroup_hadamard_transform(ushort lane_index, float lane_value) {
  for (ushort stride = 1; stride < HADAMARD_TRANSFORM_BLOCK_SIZE; stride <<= 1) {
    float partner_lane_value = simd_shuffle_xor(lane_value, stride);
    lane_value = (lane_index & stride) ? (partner_lane_value - lane_value) : (partner_lane_value + lane_value);
  }

  return lane_value / sqrt(static_cast<float>(HADAMARD_TRANSFORM_BLOCK_SIZE));
}

template <typename T>
static METAL_FUNC T simdgroup_input_random_hadamard_transform(ushort lane_index, T lane_value, int32_t lane_factor) {
  return T(simdgroup_hadamard_transform(lane_index, float(lane_value) * float(lane_factor)));
}

template <typename T>
static METAL_FUNC T simdgroup_output_random_hadamard_transform(ushort lane_index, T lane_value, int32_t lane_factor) {
  return T(simdgroup_hadamard_transform(lane_index, float(lane_value)) * float(lane_factor));
}

// Each thread owns HADAMARD_VECTOR_SIZE consecutive elements;
#define HADAMARD_VECTOR_SIZE 4
UZU_CONST ushort HADAMARD_LANES_PER_BLOCK = HADAMARD_TRANSFORM_BLOCK_SIZE / HADAMARD_VECTOR_SIZE;

static METAL_FUNC float4 simdgroup_hadamard_transform_vector(ushort lane_index, float4 lane_values) {
  for (ushort stride = 1; stride < HADAMARD_VECTOR_SIZE; stride <<= 1) {
    METAL_PRAGMA_UNROLL
    for (ushort index = 0; index < HADAMARD_VECTOR_SIZE; ++index) {
      if (!(index & stride)) {
        const float value = lane_values[index];
        const float partner_value = lane_values[index + stride];
        lane_values[index] = partner_value + value;
        lane_values[index + stride] = value - partner_value;
      }
    }
  }
  for (ushort lane_stride = 1; lane_stride < HADAMARD_LANES_PER_BLOCK; lane_stride <<= 1) {
    const float4 partner_values = simd_shuffle_xor(lane_values, lane_stride);
    lane_values = (lane_index & lane_stride) ? (partner_values - lane_values) : (partner_values + lane_values);
  }
  return lane_values / sqrt(static_cast<float>(HADAMARD_TRANSFORM_BLOCK_SIZE));
}

template <typename T>
static METAL_FUNC vec<T, HADAMARD_VECTOR_SIZE> load_hadamard_vector(const device T* source) {
  return *reinterpret_cast<const device packed_vec<T, HADAMARD_VECTOR_SIZE>*>(source);
}

template <typename T, typename SourceT>
static METAL_FUNC void store_hadamard_vector(device T* destination, const vec<SourceT, HADAMARD_VECTOR_SIZE> source) {
  *reinterpret_cast<device packed_vec<T, HADAMARD_VECTOR_SIZE>*>(destination) = vec<T, HADAMARD_VECTOR_SIZE>(source);
}
