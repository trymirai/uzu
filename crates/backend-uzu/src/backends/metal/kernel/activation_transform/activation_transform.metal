#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../generated/activation_transform.h"
#include "../hadamard_transform/hadamard_transform.h"

using namespace metal;
using namespace uzu::activation_transform;

UZU_CONST float SYM_QMAX = 127.0;

template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(ActivationTransform)(
    const device T* input OPTIONAL(!in_place),
    device T* fp_out OPTIONAL(ops == ActivationTransformOp::InputRht || ops == ActivationTransformOp::OutputRht),
    device int8_t* q_out OPTIONAL(ops == ActivationTransformOp::Quantize || ops == ActivationTransformOp::QuantizeWithGroupSums),
    device float* scales_out OPTIONAL(ops == ActivationTransformOp::Quantize || ops == ActivationTransformOp::QuantizeWithGroupSums),
    device int32_t* group_sums_out OPTIONAL(ops == ActivationTransformOp::QuantizeWithGroupSums),
    const device int32_t* rht_factors,
    constant uint& batch_size,
    constant uint& element_count,
    const ActivationTransformOp ops SPECIALIZE,
    const bool in_place SPECIALIZE,
    uint block_index GROUPS(element_count.div_ceil(METAL_SIMD_SIZE)),
    uint batch_index GROUPS(batch_size),
    uint lane_index THREADS(METAL_SIMD_SIZE)
) {
  if (in_place) {
    input = reinterpret_cast<const device T*>(fp_out);
  }

  const bool input_rht = ops != ActivationTransformOp::OutputRht;
  const bool quantize = ops == ActivationTransformOp::Quantize || ops == ActivationTransformOp::QuantizeWithGroupSums;
  const uint factor_index = block_index * METAL_SIMD_SIZE + lane_index;
  const uint element_index = batch_index * element_count + factor_index;

  float value = static_cast<float>(input[element_index]);
  if (input_rht) {
    value = simdgroup_input_random_hadamard_transform(lane_index, value, rht_factors[factor_index]);
  } else {
    value = simdgroup_output_random_hadamard_transform(lane_index, value, rht_factors[factor_index]);
  }

  if (quantize) {
    const float magnitude = max(fabs(simd_min(value)), fabs(simd_max(value)));
    const float scale = isfinite(magnitude) && magnitude > 0.0f ? magnitude / SYM_QMAX : 1.0f;

    const int8_t code = static_cast<int8_t>(clamp(round(value / scale), -SYM_QMAX, SYM_QMAX));
    q_out[element_index] = code;

    const uint group_index = batch_index * (element_count / METAL_SIMD_SIZE) + block_index;
    if (lane_index == 0) {
      scales_out[group_index] = scale;
    }

    if (ops == ActivationTransformOp::QuantizeWithGroupSums) {
      const int group_sum = simd_sum(int(code));
      if (lane_index == 0) {
        group_sums_out[group_index] = group_sum;
      }
    }
  } else {
    fp_out[element_index] = static_cast<T>(value);
  }
}
