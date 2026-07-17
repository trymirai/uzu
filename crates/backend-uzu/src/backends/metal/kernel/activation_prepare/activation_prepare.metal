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

  const bool asymmetric = ops.contains(ActivationPrepareOps::ASYMMETRIC);
  const bool emit_row_sums = ops.contains(ActivationPrepareOps::ROW_SUMS);
  const uint row_offset = batch_idx * element_count;
  const device InputT* row_input = input + row_offset;
  device int8_t* row_q = q_out + row_offset;
  const uint groups = (element_count + group_size - 1u) / group_size;
  device float* row_scales = scales_out + batch_idx * groups;
  device int32_t* row_sums = emit_row_sums ? (row_sums_out + batch_idx * groups) : nullptr;
  device int8_t* row_zero_points = asymmetric ? (zero_points_out + batch_idx * groups) : nullptr;

  for (uint group = 0; group < groups; ++group) {
    const uint start = group * group_size;
    const uint end = min(start + group_size, element_count);
    const uint count = max(end - start, 1u);

    float thread_absmax = 0.0f;
    float thread_min = INFINITY;
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;
    float thread_sum_of_squares = 0.0f;
    for (uint i = start + thread_in_row; i < end; i += ACTIVATION_PREPARE_BLOCK_SIZE) {
      float value = static_cast<float>(row_input[i]);
      if (ops.contains(ActivationPrepareOps::INPUT_RHT)) {
        value = simdgroup_input_random_hadamard_transform(
            static_cast<ushort>(i % HADAMARD_TRANSFORM_BLOCK_SIZE), value, rht_factors[i]);
      }
      thread_absmax = max(thread_absmax, fabs(value));
      thread_min = min(thread_min, value);
      thread_max = max(thread_max, value);
      thread_sum += value;
      thread_sum_of_squares += value * value;
    }

    float scale = 1.0f;
    int8_t zero_point = 0;
    if (asymmetric) {
      if (stat == ActivationScaleStatistic::AbsMax) {
        const float min_v = threadgroup_cooperative_reduce<SimdReduceMin<float>, ACTIVATION_PREPARE_BLOCK_SIZE>(
            thread_min, shared_reduce, thread_context);
        const float max_v = threadgroup_cooperative_reduce<SimdReduceMax<float>, ACTIVATION_PREPARE_BLOCK_SIZE>(
            thread_max, shared_reduce, thread_context);
        if (isfinite(min_v) && isfinite(max_v) && max_v > min_v) {
          const float qmin = -INT8_ASYMMETRIC_QUANTIZATION_MINIMUM_MAGNITUDE;
          scale = (max_v - min_v) / (INT8_ASYMMETRIC_QUANTIZATION_MAXIMUM - qmin);
          scale = max(scale, metal::numeric_limits<float>::epsilon());
          const float zp_f = clamp(
              round(qmin - min_v / scale),
              qmin,
              INT8_ASYMMETRIC_QUANTIZATION_MAXIMUM);
          zero_point = static_cast<int8_t>(zp_f);
        }
      } else {
        const float total = threadgroup_cooperative_reduce<SimdReduceSum<float>, ACTIVATION_PREPARE_BLOCK_SIZE>(
            thread_sum, shared_reduce, thread_context);
        const float total_sq = threadgroup_cooperative_reduce<SimdReduceSum<float>, ACTIVATION_PREPARE_BLOCK_SIZE>(
            thread_sum_of_squares, shared_reduce, thread_context);
        const float mean = total / static_cast<float>(count);
        const float rms = sqrt(total_sq / static_cast<float>(count));
        scale = rms > 0.0f ? rms / INT8_SYMMETRIC_QUANTIZATION_MAXIMUM : 1.0f;
        const float qmin = -INT8_ASYMMETRIC_QUANTIZATION_MINIMUM_MAGNITUDE;
        const float zp_f = clamp(round(-mean / scale), qmin, INT8_ASYMMETRIC_QUANTIZATION_MAXIMUM);
        zero_point = static_cast<int8_t>(zp_f);
      }
      if (thread_in_row == 0) {
        row_scales[group] = scale;
        row_zero_points[group] = zero_point;
      }
    } else {
      float scale_stat;
      if (stat == ActivationScaleStatistic::AbsMax) {
        scale_stat = threadgroup_cooperative_reduce<SimdReduceMax<float>, ACTIVATION_PREPARE_BLOCK_SIZE>(
            thread_absmax, shared_reduce, thread_context);
      } else {
        const float total_sq = threadgroup_cooperative_reduce<SimdReduceSum<float>, ACTIVATION_PREPARE_BLOCK_SIZE>(
            thread_sum_of_squares, shared_reduce, thread_context);
        scale_stat = sqrt(total_sq / static_cast<float>(count));
      }
      scale = scale_stat > 0.0f ? scale_stat / INT8_SYMMETRIC_QUANTIZATION_MAXIMUM : 1.0f;
      if (thread_in_row == 0) {
        row_scales[group] = scale;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    scale = row_scales[group];
    if (asymmetric) {
      zero_point = row_zero_points[group];
    }

    int thread_row_sum = 0;
    for (uint i = start + thread_in_row; i < end; i += ACTIVATION_PREPARE_BLOCK_SIZE) {
      float value = static_cast<float>(row_input[i]);
      if (ops.contains(ActivationPrepareOps::INPUT_RHT)) {
        value = simdgroup_input_random_hadamard_transform(
            static_cast<ushort>(i % HADAMARD_TRANSFORM_BLOCK_SIZE), value, rht_factors[i]);
      }
      int8_t q;
      if (asymmetric) {
        const float qmin = -INT8_ASYMMETRIC_QUANTIZATION_MINIMUM_MAGNITUDE;
        q = static_cast<int8_t>(clamp(
            round(value / scale) + static_cast<float>(zero_point),
            qmin,
            INT8_ASYMMETRIC_QUANTIZATION_MAXIMUM));
      } else {
        q = static_cast<int8_t>(clamp(
            round(value / scale),
            -INT8_SYMMETRIC_QUANTIZATION_MAXIMUM,
            INT8_SYMMETRIC_QUANTIZATION_MAXIMUM));
      }
      row_q[i] = q;
      if (emit_row_sums) {
        thread_row_sum += static_cast<int>(q);
      }
    }

    if (emit_row_sums) {
      const int group_sum = static_cast<int>(threadgroup_cooperative_reduce<
          SimdReduceSum<float>,
          ACTIVATION_PREPARE_BLOCK_SIZE>(static_cast<float>(thread_row_sum), shared_reduce, thread_context));
      if (thread_in_row == 0) {
        row_sums[group] = group_sum;
      }
    }
  }
}
