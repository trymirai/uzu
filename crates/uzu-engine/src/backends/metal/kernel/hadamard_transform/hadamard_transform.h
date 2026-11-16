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

template <typename T>
static METAL_FUNC vec<T, 4> load_vec4(const device T* source) {
  return *reinterpret_cast<const device packed_vec<T, 4>*>(source);
}

template <typename T, typename SourceT>
static METAL_FUNC void store_vec4(device T* destination, const vec<SourceT, 4> source) {
  *reinterpret_cast<device packed_vec<T, 4>*>(destination) = vec<T, 4>(source);
}

// Preprocessor form: host codegen expands it inside GROUPS(...) expressions, which cannot see constexprs.
#define HADAMARD_VECTOR_SIZE 4

namespace hadamard {

/// Normalized Walsh-Hadamard transform of `VALUES_PER_LANE` values in each of `LANES` lanes.
template <ushort VALUES_PER_LANE, ushort LANES>
static METAL_FUNC void simdgroup_transform(ushort lane_index, thread float* values) {
  constexpr ushort SIZE = VALUES_PER_LANE * LANES;
  constexpr ushort CHUNK = VALUES_PER_LANE < 4 ? VALUES_PER_LANE : 4;
  using Chunk = metal::conditional_t<CHUNK == 1, float, vec<float, CHUNK>>;
  static_assert(
      LANES <= METAL_SIMD_SIZE && (LANES & (LANES - 1)) == 0,
      "lanes must be a power of two within a simdgroup"
  );
  static_assert(
      SIZE == 32 || SIZE == 64 || SIZE == 128 || SIZE == 256 || SIZE == 512,
      "transform size must be 32, 64, 128, 256 or 512"
  );
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

static METAL_FUNC float simdgroup_transform(ushort lane_index, float lane_value) {
  simdgroup_transform<1, HADAMARD_TRANSFORM_BLOCK_SIZE>(lane_index, &lane_value);
  return lane_value;
}

template <typename T>
static METAL_FUNC T simdgroup_output_transform(ushort lane_index, T lane_value, int32_t lane_factor) {
  return T(simdgroup_transform(lane_index, float(lane_value)) * float(lane_factor));
}

// Each thread owns VECTOR_SIZE consecutive elements.
UZU_CONST ushort VECTOR_SIZE = HADAMARD_VECTOR_SIZE;
UZU_CONST ushort LANES_PER_BLOCK = HADAMARD_TRANSFORM_BLOCK_SIZE / VECTOR_SIZE;

static METAL_FUNC float4 simdgroup_transform_vector(ushort lane_index, float4 lane_values) {
  simdgroup_transform<VECTOR_SIZE, LANES_PER_BLOCK>(lane_index, reinterpret_cast<thread float*>(&lane_values));
  return lane_values;
}

} // namespace hadamard
