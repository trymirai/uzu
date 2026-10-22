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

/// Normalized Walsh-Hadamard transform of `VALUES_PER_LANE` values in each of `LANES` lanes.
template <ushort VALUES_PER_LANE, ushort LANES>
static METAL_FUNC void simdgroup_hadamard_transform(ushort lane_index, thread float* values) {
  constexpr ushort SIZE = VALUES_PER_LANE * LANES;
  constexpr ushort CHUNK = VALUES_PER_LANE < 4 ? VALUES_PER_LANE : 4;
  using Chunk = metal::conditional_t<CHUNK == 1, float, vec<float, CHUNK>>;
  static_assert(
      LANES <= METAL_SIMD_SIZE && (LANES & (LANES - 1)) == 0,
      "lanes must be a power of two within a simdgroup"
  );
  static_assert(SIZE == 32 || SIZE == 64 || SIZE == 128 || SIZE == 256, "transform size must be 32, 64, 128 or 256");
  METAL_PRAGMA_UNROLL
  for (ushort stride = 1; stride < VALUES_PER_LANE; stride <<= 1) {
    METAL_PRAGMA_UNROLL
    for (ushort index = 0; index < VALUES_PER_LANE; ++index) {
      if (!(index & stride)) {
        const float value = values[index];
        const float partner_value = values[index + stride];
        values[index] = partner_value + value;
        values[index + stride] = value - partner_value;
      }
    }
  }
  METAL_PRAGMA_UNROLL
  for (ushort lane_stride = 1; lane_stride < LANES; lane_stride <<= 1) {
    METAL_PRAGMA_UNROLL
    for (ushort index = 0; index < VALUES_PER_LANE; index += CHUNK) {
      thread Chunk& chunk = *reinterpret_cast<thread Chunk*>(values + index);
      const Chunk partner = simd_shuffle_xor(chunk, lane_stride);
      chunk = (lane_index & lane_stride) ? (partner - chunk) : (partner + chunk);
    }
  }
  METAL_PRAGMA_UNROLL
  for (ushort index = 0; index < VALUES_PER_LANE; ++index) {
    values[index] /= sqrt(static_cast<float>(SIZE));
  }
}

static METAL_FUNC float simdgroup_hadamard_transform(ushort lane_index, float lane_value) {
  simdgroup_hadamard_transform<1, HADAMARD_TRANSFORM_BLOCK_SIZE>(lane_index, &lane_value);
  return lane_value;
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
  simdgroup_hadamard_transform<HADAMARD_VECTOR_SIZE, HADAMARD_LANES_PER_BLOCK>(
      lane_index,
      reinterpret_cast<thread float*>(&lane_values)
  );
  return lane_values;
}

template <typename T>
static METAL_FUNC vec<T, HADAMARD_VECTOR_SIZE> load_hadamard_vector(const device T* source) {
  return *reinterpret_cast<const device packed_vec<T, HADAMARD_VECTOR_SIZE>*>(source);
}

template <typename T, typename SourceT>
static METAL_FUNC void store_hadamard_vector(device T* destination, const vec<SourceT, HADAMARD_VECTOR_SIZE> source) {
  *reinterpret_cast<device packed_vec<T, HADAMARD_VECTOR_SIZE>*>(destination) = vec<T, HADAMARD_VECTOR_SIZE>(source);
}
