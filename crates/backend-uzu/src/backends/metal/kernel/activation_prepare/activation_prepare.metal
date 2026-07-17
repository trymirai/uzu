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
METAL_CONST uint BLOCK_SIZE = ACTIVATION_PREPARE_BLOCK_SIZE;
METAL_CONST float ASYM_QMIN = -INT8_ASYMMETRIC_QUANTIZATION_MINIMUM_MAGNITUDE;
METAL_CONST float ASYM_QMAX = INT8_ASYMMETRIC_QUANTIZATION_MAXIMUM;
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
    device int8_t* zero_points_out OPTIONAL(ops.contains(ActivationPrepareOps::ASYMMETRIC)),
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
  const uint groups = div_ceil(element_count, group_size);
  device float* row_scales = scales_out + batch_idx * groups;

  for (uint group = 0; group < groups; ++group) {
    const uint start = group * group_size;
    const uint end = min(start + group_size, element_count);
    const float count = static_cast<float>(max(end - start, 1u));

    float thread_absmax = 0.0f;
    float thread_min = INFINITY;
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;
    float thread_sum_of_squares = 0.0f;
    METAL_PRAGMA_UNROLL
    for (uint i = start + thread_in_row; i < end; i += BLOCK_SIZE) {
      const float value = load_activation(row_input, i, rht_factors, ops);
      if (ops.contains(ActivationPrepareOps::ASYMMETRIC)) {
        if (stat == ActivationScaleStatistic::AbsMax) {
          thread_min = min(thread_min, value);
          thread_max = max(thread_max, value);
        } else {
          thread_sum += value;
          thread_sum_of_squares += value * value;
        }
      } else if (stat == ActivationScaleStatistic::AbsMax) {
        thread_absmax = max(thread_absmax, fabs(value));
      } else {
        thread_sum_of_squares += value * value;
      }
    }

    float scale = 1.0f;
    int8_t zero_point = 0;
    if (ops.contains(ActivationPrepareOps::ASYMMETRIC)) {
      if (stat == ActivationScaleStatistic::AbsMax) {
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
        if (isfinite(min_v) && isfinite(max_v) && max_v > min_v) {
          scale = max((max_v - min_v) / (ASYM_QMAX - ASYM_QMIN), metal::numeric_limits<float>::epsilon());
          zero_point = static_cast<int8_t>(clamp(round(ASYM_QMIN - min_v / scale), ASYM_QMIN, ASYM_QMAX));
        }
      } else {
        const float total = threadgroup_cooperative_reduce<SimdReduceSum<float>, ACTIVATION_PREPARE_BLOCK_SIZE>(
            thread_sum,
            shared_reduce,
            thread_context
        );
        const float total_sq = threadgroup_cooperative_reduce<SimdReduceSum<float>, ACTIVATION_PREPARE_BLOCK_SIZE>(
            thread_sum_of_squares,
            shared_reduce,
            thread_context
        );
        const float rms = sqrt(total_sq / count);
        scale = rms > 0.0f ? rms / SYM_QMAX : 1.0f;
        zero_point = static_cast<int8_t>(clamp(round(-(total / count) / scale), ASYM_QMIN, ASYM_QMAX));
      }
      if (thread_in_row == 0) {
        row_scales[group] = scale;
        zero_points_out[batch_idx * groups + group] = zero_point;
      }
    } else {
      const float scale_stat =
          stat == ActivationScaleStatistic::AbsMax
              ? threadgroup_cooperative_reduce<SimdReduceMax<float>, ACTIVATION_PREPARE_BLOCK_SIZE>(
                    thread_absmax,
                    shared_reduce,
                    thread_context
                )
              : sqrt(
                    threadgroup_cooperative_reduce<SimdReduceSum<float>, ACTIVATION_PREPARE_BLOCK_SIZE>(
                        thread_sum_of_squares,
                        shared_reduce,
                        thread_context
                    ) /
                    count
                );
      scale = scale_stat > 0.0f ? scale_stat / SYM_QMAX : 1.0f;
      if (thread_in_row == 0) {
        row_scales[group] = scale;
      }
    }

    // All threads already hold identical scale/zp after the cooperative reduce broadcast.
    const float inv_scale = 1.0f / scale;
    int thread_row_sum = 0;
    METAL_PRAGMA_UNROLL
    for (uint i = start + thread_in_row; i < end; i += BLOCK_SIZE) {
      const float value = load_activation(row_input, i, rht_factors, ops);
      int8_t q;
      if (ops.contains(ActivationPrepareOps::ASYMMETRIC)) {
        q = static_cast<int8_t>(clamp(round(value * inv_scale) + static_cast<float>(zero_point), ASYM_QMIN, ASYM_QMAX));
      } else {
        q = static_cast<int8_t>(clamp(round(value * inv_scale), -SYM_QMAX, SYM_QMAX));
      }
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
