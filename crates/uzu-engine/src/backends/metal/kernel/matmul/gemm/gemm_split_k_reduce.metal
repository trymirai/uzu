#include "../../common/dsl.h"
#include "../../common/thread_context.h"
#include "../generated/gemm.h"
#include "../../hadamard_transform/hadamard_transform.h"

using namespace metal;
using namespace uzu::gemm;

template <typename Input, typename T>
VARIANTS(Input, float, half, bfloat)
VARIANTS(T, float, half, bfloat)
CONSTRAINT(Input == "float" || Input == T)
KERNEL(GemmSplitKReduce)(
    const device Input* partial_sums,
    device T* output,
    const device T* output_bias
        OPTIONAL(output_transform.contains(GemmDTransform::BIAS)),
    const device int32_t* rht_factors
        OPTIONAL(output_transform.contains(GemmDTransform::RHT)),
    const constant uint& element_count,
    const constant uint& partition_count,
    const constant uint& threadgroup_count,
    const constant uint& column_count,
    const constant float& output_scale
        OPTIONAL(output_transform.contains(GemmDTransform::SCALE)),
    const GemmDTransform output_transform SPECIALIZE,
    const uint threadgroup_index GROUPS(threadgroup_count),
    const uint thread_index_in_threadgroup THREADS(256),
    const ThreadContext thread_context
) {
  (void)threadgroup_index;
  (void)thread_index_in_threadgroup;

  const uint threads_per_threadgroup = thread_context.simdgroups_per_threadgroup * thread_context.simdgroup_size;
  const uint local_thread_index =
      thread_context.simdgroup_index * thread_context.simdgroup_size + thread_context.simd_lane_id;
  const uint vector_count = element_count / 4u;
  const uint vector_index = thread_context.threadgroup_position.x * threads_per_threadgroup + local_thread_index;
  if (vector_index >= vector_count) {
    return;
  }

  device vec<T, 4>* output_vectors = reinterpret_cast<device vec<T, 4>*>(output);
  const device vec<Input, 4>* partial_sum_vectors = reinterpret_cast<const device vec<Input, 4>*>(partial_sums);

  float4 accumulator = float4(0.0f);
  for (uint partition = 0u; partition < partition_count; ++partition) {
    accumulator += float4(partial_sum_vectors[partition * vector_count + vector_index]);
  }

  if (output_transform.contains(GemmDTransform::SCALE)) {
    accumulator *= output_scale;
  }
  if (output_transform.contains(GemmDTransform::ACCUMULATE)) {
    accumulator += float4(output_vectors[vector_index]);
  }
  if (output_transform.contains(GemmDTransform::BIAS)) {
    const uint column = (vector_index * 4u) % column_count;
    accumulator += float4(*reinterpret_cast<const device vec<T, 4>*>(output_bias + column));
  }

  if (output_transform.contains(GemmDTransform::RHT)) {
    // The output RHT of ActivationTransform, from the same bf16 output and in the same stage order, so the result is
    // bitwise identical. A 32-wide block is 8 lanes x 4 values: element bits 0-1 sit in registers, bits 2-4 in lanes.
    float4 value = float4(vec<T, 4>(accumulator));
    value = float4(value.y + value.x, value.x - value.y, value.w + value.z, value.z - value.w);
    value = float4(value.z + value.x, value.w + value.y, value.x - value.z, value.y - value.w);
    const ushort lane = thread_context.simd_lane_id;
    for (ushort stride = 1; stride < HADAMARD_TRANSFORM_BLOCK_SIZE / 4; stride <<= 1) {
      const float4 partner = simd_shuffle_xor(value, stride);
      value = (lane & stride) ? (partner - value) : (partner + value);
    }
    const int4 factors = *reinterpret_cast<const device int4*>(rht_factors + (vector_index * 4u) % column_count);
    accumulator = value / sqrt(static_cast<float>(HADAMARD_TRANSFORM_BLOCK_SIZE)) * float4(factors);
  }

  output_vectors[vector_index] = vec<T, 4>(accumulator);
}
