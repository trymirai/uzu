#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../generated/activation_prepare.h"
#include "../hadamard_transform/hadamard_transform.h"

using namespace metal;
using namespace uzu::activation_prepare;

static_assert(
    ACTIVATION_QUANTIZATION_GROUP_SIZE == METAL_SIMD_SIZE,
    "Activation quantization group size must match the Metal SIMD width"
);

METAL_CONST float SYM_QMAX = INT8_SYMMETRIC_QUANTIZATION_MAXIMUM;

template <typename InputT>
VARIANTS(InputT, float, bfloat)
PUBLIC KERNEL(ActivationsPrepare)(
    const device InputT* input,
    device int8_t* q_out,
    device float* scales_out,
    const device int32_t* rht_factors OPTIONAL(ops.contains(ActivationPrepareOps::INPUT_RHT)),
    constant uint& batch_size,
    constant uint& element_count,
    constant uint& group_size,
    const ActivationPrepareOps ops SPECIALIZE,
    uint block_index GROUPS(element_count.div_ceil(METAL_SIMD_SIZE)),
    uint batch_index GROUPS(batch_size),
    uint lane_index THREADS(METAL_SIMD_SIZE)
) {
  if (group_size != ACTIVATION_QUANTIZATION_GROUP_SIZE || element_count % group_size != 0) {
    return;
  }

  const uint factor_index = block_index * METAL_SIMD_SIZE + lane_index;
  const uint element_index = batch_index * element_count + factor_index;

  float value = static_cast<float>(input[element_index]);
  if (ops.contains(ActivationPrepareOps::INPUT_RHT)) {
    value = simdgroup_input_random_hadamard_transform(lane_index, value, rht_factors[factor_index]);
  }

  const float magnitude = max(fabs(simd_min(value)), fabs(simd_max(value)));
  const float scale = isfinite(magnitude) && magnitude > 0.0f ? magnitude / SYM_QMAX : 1.0f;
  if (lane_index == 0) {
    scales_out[batch_index * (element_count / group_size) + block_index] = scale;
  }

  q_out[element_index] = static_cast<int8_t>(clamp(round(value / scale), -SYM_QMAX, SYM_QMAX));
}
