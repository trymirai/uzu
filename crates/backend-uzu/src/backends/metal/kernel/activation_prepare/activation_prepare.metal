#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"
#include "../generated/activation_prepare.h"
#include "../hadamard_transform/hadamard_transform.h"

using namespace metal;
using namespace uzu::activation_prepare;

static_assert(
    ACTIVATION_QUANTIZATION_GROUP_SIZE == METAL_SIMD_SIZE,
    "Activation quantization group size must match the Metal SIMD width"
);

#define ACTIVATION_PREPARE_BLOCK_SIZE 256
METAL_CONST uint BLOCK_SIZE = ACTIVATION_PREPARE_BLOCK_SIZE;
METAL_CONST float SYM_QMAX = INT8_SYMMETRIC_QUANTIZATION_MAXIMUM;

template <typename InputT>
METAL_FUNC float load_activation(
    const device InputT* row_input,
    uint index,
    const device int32_t* rht_factors,
    ActivationPrepareOps ops
) {
  float value = static_cast<float>(row_input[index]);
  if (ops.contains(ActivationPrepareOps::INPUT_RHT)) {
    value = simdgroup_input_random_hadamard_transform(
        static_cast<ushort>(index % HADAMARD_TRANSFORM_BLOCK_SIZE),
        value,
        rht_factors[index]
    );
  }
  return value;
}

template <typename InputT>
VARIANTS(InputT, float, half, bfloat)
PUBLIC KERNEL(ActivationsPrepare)(
    const device InputT* input,
    device int8_t* q_out OPTIONAL(ops.contains(ActivationPrepareOps::QUANTIZE)),
    device float* scales_out OPTIONAL(ops.contains(ActivationPrepareOps::QUANTIZE)),
    device int32_t* row_sums_out OPTIONAL(ops.contains(ActivationPrepareOps::ROW_SUMS)),
    const device int32_t* rht_factors OPTIONAL(ops.contains(ActivationPrepareOps::INPUT_RHT)),
    constant uint& batch_size,
    constant uint& element_count,
    constant uint& group_size,
    const ActivationPrepareOps ops SPECIALIZE,
    threadgroup float shared_reduce[METAL_SIMD_SIZE],
    const ThreadContext thread_context,
    const uint batch_idx GROUPS(batch_size),
    const uint thread_in_row THREADS(ACTIVATION_PREPARE_BLOCK_SIZE)
) {
  if (!ops.contains(ActivationPrepareOps::QUANTIZE) || group_size != ACTIVATION_QUANTIZATION_GROUP_SIZE) {
    return;
  }

  const uint row_offset = batch_idx * element_count;
  const device InputT* row_input = input + row_offset;
  device int8_t* row_q = q_out + row_offset;
  const uint groups = div_ceil(element_count, group_size);
  device float* row_scales = scales_out + batch_idx * groups;

  for (uint group = 0; group < groups; ++group) {
    const uint start = group * group_size;
    const uint end = min(start + group_size, element_count);
    float thread_min = INFINITY;
    float thread_max = -INFINITY;
    METAL_PRAGMA_UNROLL
    for (uint i = start + thread_in_row; i < end; i += BLOCK_SIZE) {
      const float value = load_activation(row_input, i, rht_factors, ops);
      thread_min = min(thread_min, value);
      thread_max = max(thread_max, value);
    }

    const float min_v = threadgroup_cooperative_reduce<SimdReduceMin<float>, ACTIVATION_PREPARE_BLOCK_SIZE>(
        thread_min,
        shared_reduce,
        thread_context
    );
    const float max_v = threadgroup_cooperative_reduce<SimdReduceMax<float>, ACTIVATION_PREPARE_BLOCK_SIZE>(
        thread_max,
        shared_reduce,
        thread_context
    );
    const float magnitude = max(fabs(min_v), fabs(max_v));
    const float scale = isfinite(magnitude) && magnitude > 0.0f ? magnitude / SYM_QMAX : 1.0f;
    if (thread_in_row == 0) {
      row_scales[group] = scale;
    }

    // All threads already hold the same scale after the cooperative reduce broadcast.
    const float inv_scale = 1.0f / scale;
    int thread_row_sum = 0;
    METAL_PRAGMA_UNROLL
    for (uint i = start + thread_in_row; i < end; i += BLOCK_SIZE) {
      const float value = load_activation(row_input, i, rht_factors, ops);
      const int8_t q = static_cast<int8_t>(clamp(round(value * inv_scale), -SYM_QMAX, SYM_QMAX));
      row_q[i] = q;
      if (ops.contains(ActivationPrepareOps::ROW_SUMS)) {
        thread_row_sum += static_cast<int>(q);
      }
    }

    if (ops.contains(ActivationPrepareOps::ROW_SUMS)) {
      const int group_sum =
          static_cast<int>(threadgroup_cooperative_reduce<SimdReduceSum<float>, ACTIVATION_PREPARE_BLOCK_SIZE>(
              static_cast<float>(thread_row_sum),
              shared_reduce,
              thread_context
          ));
      if (thread_in_row == 0) {
        row_sums_out[batch_idx * groups + group] = group_sum;
      }
    }
  }
}
