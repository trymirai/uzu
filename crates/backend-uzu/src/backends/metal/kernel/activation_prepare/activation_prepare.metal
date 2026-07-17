#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"
#include "../generated/activation_prepare.h"
#include "../hadamard_transform/hadamard_transform.h"

using namespace metal;
using namespace uzu::activation_prepare;

#define ACTIVATION_PREPARE_BLOCK_SIZE 256
#define INT8_SYMMETRIC_QMAX 127.0f

template <typename InputT>
VARIANTS(InputT, float, half, bfloat)
PUBLIC KERNEL(ActivationsPrepare)(
    const device InputT* input,
    device int8_t* q_out OPTIONAL(ops.contains(ActivationPrepareOps::QUANTIZE)),
    device float* scales_out OPTIONAL(ops.contains(ActivationPrepareOps::QUANTIZE)),
    const device int32_t* rht_factors OPTIONAL(ops.contains(ActivationPrepareOps::INPUT_RHT)),
    constant uint& batch_size,
    constant uint& element_count,
    constant uint& group_size,
    const ActivationPrepareOps ops SPECIALIZE,
    const ActivationScaleStatistic stat SPECIALIZE,
    threadgroup float shared_reduce[METAL_SIMD_SIZE],
    const ThreadContext thread_context,
    const uint batch_idx GROUPS(batch_size),
    const uint thread_in_row THREADS(ACTIVATION_PREPARE_BLOCK_SIZE)
) {
  if (!ops.contains(ActivationPrepareOps::QUANTIZE)) {
    return;
  }

  const uint row_offset = batch_idx * element_count;
  const device InputT* row_input = input + row_offset;
  device int8_t* row_q = q_out + row_offset;
  const uint groups = (element_count + group_size - 1u) / group_size;
  device float* row_scales = scales_out + batch_idx * groups;

  for (uint group = 0; group < groups; ++group) {
    const uint start = group * group_size;
    const uint end = min(start + group_size, element_count);
    float thread_absmax = 0.0f;
    float thread_sum_of_squares = 0.0f;
    for (uint i = start + thread_in_row; i < end; i += ACTIVATION_PREPARE_BLOCK_SIZE) {
      float value = static_cast<float>(row_input[i]);
      if (ops.contains(ActivationPrepareOps::INPUT_RHT)) {
        value = simdgroup_input_random_hadamard_transform(
            static_cast<ushort>(i % HADAMARD_TRANSFORM_BLOCK_SIZE), value, rht_factors[i]);
      }
      thread_absmax = max(thread_absmax, fabs(value));
      thread_sum_of_squares += value * value;
    }

    float scale_stat;
    if (stat == ActivationScaleStatistic::AbsMax) {
      scale_stat = threadgroup_cooperative_reduce<SimdReduceMax<float>, ACTIVATION_PREPARE_BLOCK_SIZE>(
          thread_absmax, shared_reduce, thread_context);
    } else {
      const float total_sq = threadgroup_cooperative_reduce<SimdReduceSum<float>, ACTIVATION_PREPARE_BLOCK_SIZE>(
          thread_sum_of_squares, shared_reduce, thread_context);
      scale_stat = sqrt(total_sq / static_cast<float>(max(end - start, 1u)));
    }
    const float divisor = scale_stat > 0.0f ? scale_stat / INT8_SYMMETRIC_QMAX : 1.0f;
    if (thread_in_row == 0) {
      row_scales[group] = divisor;
    }

    for (uint i = start + thread_in_row; i < end; i += ACTIVATION_PREPARE_BLOCK_SIZE) {
      float value = static_cast<float>(row_input[i]);
      if (ops.contains(ActivationPrepareOps::INPUT_RHT)) {
        value = simdgroup_input_random_hadamard_transform(
            static_cast<ushort>(i % HADAMARD_TRANSFORM_BLOCK_SIZE), value, rht_factors[i]);
      }
      row_q[i] = static_cast<int8_t>(clamp(round(value / divisor), -INT8_SYMMETRIC_QMAX, INT8_SYMMETRIC_QMAX));
    }
  }
}
