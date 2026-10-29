#include "../../common/dsl.h"
#include "../../common/soft_cap.h"
#include "../../common/thread_context.h"
#include "../generated/gemm.h"

using namespace metal;
using namespace uzu::gemm;

template <typename Input, typename T>
VARIANTS(Input, float, bfloat)
VARIANTS(T, float, bfloat)
CONSTRAINT(Input == "float" || Input == T)
KERNEL(GemmSplitKReduce)(
    const device Input* partial_sums,
    device T* output,
    const device T* output_bias
        OPTIONAL(output_transform.contains(GemmDTransform::BIAS)),
    const constant uint& element_count,
    const constant uint& partition_count,
    const constant uint& threadgroup_count,
    const constant uint& column_count,
    const constant float& output_scale
        OPTIONAL(output_transform.contains(GemmDTransform::SCALE)),
    const constant float& soft_cap
        OPTIONAL(output_transform.contains(GemmDTransform::SOFT_CAP)),
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
  if (output_transform.contains(GemmDTransform::BIAS)) {
    const uint column = (vector_index * 4u) % column_count;
    accumulator += float4(*reinterpret_cast<const device vec<T, 4>*>(output_bias + column));
  }
  if (output_transform.contains(GemmDTransform::SOFT_CAP)) {
    accumulator = uzu::apply_soft_cap(accumulator, soft_cap);
  }

  output_vectors[vector_index] = vec<T, 4>(accumulator);
}
