#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../hadamard_transform/hadamard_transform.h"

using namespace metal;

UZU_CONST float SYM_QMAX = 127.0;

template <typename InputT>
VARIANTS(InputT, float, bfloat)
PUBLIC KERNEL(RHTQuantizeActivations)(
    const device InputT* input,
    device int8_t* q_out,
    device float* scales_out,
    device int32_t* group_sums_out OPTIONAL(emit_group_sums),
    const device int32_t* rht_factors,
    constant uint& batch_size,
    constant uint& element_count,
    constant uint& group_size,
    const bool emit_group_sums SPECIALIZE,
    uint block_index GROUPS(element_count.div_ceil(METAL_SIMD_SIZE)),
    uint batch_index GROUPS(batch_size),
    uint lane_index THREADS(METAL_SIMD_SIZE)
) {
  const uint factor_index = block_index * METAL_SIMD_SIZE + lane_index;
  const uint element_index = batch_index * element_count + factor_index;

  float value = static_cast<float>(input[element_index]);
  value = simdgroup_input_random_hadamard_transform(lane_index, value, rht_factors[factor_index]);

  const float magnitude = max(fabs(simd_min(value)), fabs(simd_max(value)));
  const float scale = isfinite(magnitude) && magnitude > 0.0f ? magnitude / SYM_QMAX : 1.0f;

  const int8_t code = static_cast<int8_t>(clamp(round(value / scale), -SYM_QMAX, SYM_QMAX));
  q_out[element_index] = code;

  const uint group_index = batch_index * (element_count / group_size) + block_index;
  if (lane_index == 0) {
    scales_out[group_index] = scale;
  }

  if (emit_group_sums) {
    const int group_sum = simd_sum(int(code));
    if (lane_index == 0) {
      group_sums_out[group_index] = group_sum;
    }
  }
}
