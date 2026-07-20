#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../generated/activation_prepare.h"
#include "../hadamard_transform/hadamard_transform.h"

using namespace metal;
using namespace uzu::activation_prepare;

static_assert(
    ACTIVATION_QUANTIZATION_GROUP_SIZE == METAL_SIMD_SIZE,
    "Activation quantization group size must match the Metal SIMD width"
);

#define ACTIVATION_PREPARE_BLOCK_SIZE 256
#define ACTIVATION_PREPARE_SIMDGROUPS (ACTIVATION_PREPARE_BLOCK_SIZE / METAL_SIMD_SIZE)
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
    const ThreadContext thread_context,
    const uint batch_idx GROUPS(batch_size),
    const uint thread_in_row THREADS(ACTIVATION_PREPARE_BLOCK_SIZE)
) {
  if (
      !ops.contains(ActivationPrepareOps::QUANTIZE) || group_size != ACTIVATION_QUANTIZATION_GROUP_SIZE ||
      element_count % group_size != 0
  ) {
    return;
  }

  const uint groups = element_count / group_size;
  const uint row_offset = batch_idx * element_count;
  const device InputT* row_input = input + row_offset;
  device int8_t* row_q = q_out + row_offset;
  device float* row_scales = scales_out + batch_idx * groups;

  for (uint group = thread_context.simdgroup_index; group < groups; group += ACTIVATION_PREPARE_SIMDGROUPS) {
    const uint index = group * group_size + thread_context.simd_lane_id;
    const float value = load_activation(row_input, index, rht_factors, ops);

    const float min_v = simd_min(value);
    const float max_v = simd_max(value);
    const float magnitude = max(fabs(min_v), fabs(max_v));
    const float scale = isfinite(magnitude) && magnitude > 0.0f ? magnitude / SYM_QMAX : 1.0f;
    if (thread_context.simd_lane_id == 0) {
      row_scales[group] = scale;
    }

    const int8_t q = static_cast<int8_t>(clamp(round(value / scale), -SYM_QMAX, SYM_QMAX));
    row_q[index] = q;
    if (ops.contains(ActivationPrepareOps::ROW_SUMS)) {
      const int group_sum = static_cast<int>(simd_sum(static_cast<float>(q)));
      if (thread_context.simd_lane_id == 0) {
        row_sums_out[batch_idx * groups + group] = group_sum;
      }
    }
  }
}
