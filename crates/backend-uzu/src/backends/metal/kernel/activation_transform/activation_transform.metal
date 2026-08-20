#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/activation_quantization.h"
#include "../generated/activation_transform.h"
#include "../hadamard_transform/hadamard_transform.h"

using namespace metal;
using namespace uzu::activation_transform;

#define NUM_SIMDGROUPS 4
#define NUM_THREADS NUM_SIMDGROUPS* METAL_SIMD_SIZE

#define QUANTIZED (ops == ActivationTransformOp::Quantize || ops == ActivationTransformOp::QuantizeWithGroupSums)
#define EMITS_GROUP_SUMS (ops == ActivationTransformOp::QuantizeWithGroupSums)

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(ActivationTransform)(
    const device T* input OPTIONAL(!in_place),
    device T* fp_out OPTIONAL(!QUANTIZED),
    device int8_t* q_out OPTIONAL(QUANTIZED),
    device float* scales_out OPTIONAL(QUANTIZED),
    device int32_t* group_sums_out OPTIONAL(EMITS_GROUP_SUMS),
    const device int32_t* rht_factors,
    constant uint& batch_size,
    constant uint& element_count,
    const ActivationTransformOp ops SPECIALIZE,
    const bool in_place SPECIALIZE,
    const uint activation_scale_group_size SPECIALIZE,
    const uint sum_group_size SPECIALIZE,
    threadgroup float partial_max OPTIONAL(QUANTIZED && activation_scale_group_size > METAL_SIMD_SIZE)[NUM_SIMDGROUPS],
    threadgroup int partial_sums OPTIONAL(EMITS_GROUP_SUMS && sum_group_size > METAL_SIMD_SIZE)[NUM_SIMDGROUPS],
    uint activation_tile_index GROUPS(element_count.div_ceil(NUM_THREADS)),
    uint batch_index GROUPS(batch_size),
    uint thread_index THREADS(NUM_THREADS),
    const ThreadContext thread_context
) {
  (void)thread_index;
  if (in_place) {
    input = reinterpret_cast<const device T*>(fp_out);
  }

  const bool input_rht = ops != ActivationTransformOp::OutputRht;
  const ushort lane_index = thread_context.simd_lane_id;
  const ushort simdgroup_index = thread_context.simdgroup_index;
  const uint tile_offset = activation_tile_index * ACTIVATION_QUANT_TILE_SIZE;
  const uint simdgroup_offset = tile_offset + simdgroup_index * METAL_SIMD_SIZE;
  const bool in_bounds = simdgroup_offset < element_count;
  const uint factor_index = simdgroup_offset + lane_index;
  const uint element_index = batch_index * element_count + factor_index;

  float value = 0.0f;
  if (in_bounds) {
    value = static_cast<float>(input[element_index]);
    value = input_rht ? simdgroup_input_random_hadamard_transform(lane_index, value, rht_factors[factor_index])
                      : simdgroup_output_random_hadamard_transform(lane_index, value, rht_factors[factor_index]);
  }

  if (!QUANTIZED) {
    if (in_bounds) {
      fp_out[element_index] = static_cast<T>(value);
    }
    return;
  }

  const float maximum = reduce_activation_quantization_group(
      fabs(value),
      activation_scale_group_size,
      partial_max,
      thread_context,
      [](float x) { return simd_max(x); },
      [](float x, float y) { return max(x, y); }
  );
  const float scale = isfinite(maximum) && maximum > 0.0f ? maximum / ACTIVATION_QUANT_INT8_MAX : 1.0f;

  const int8_t code =
      static_cast<int8_t>(clamp(round(value / scale), -ACTIVATION_QUANT_INT8_MAX, ACTIVATION_QUANT_INT8_MAX));
  if (in_bounds) {
    q_out[element_index] = code;
  }
  write_activation_quantization_group(
      scales_out,
      scale,
      activation_scale_group_size,
      element_count,
      activation_tile_index,
      batch_index,
      thread_context
  );

  if (EMITS_GROUP_SUMS) {
    const int sum = reduce_activation_quantization_group(
        int(code),
        sum_group_size,
        partial_sums,
        thread_context,
        [](int x) { return simd_sum(x); },
        [](int x, int y) { return x + y; }
    );
    write_activation_quantization_group(
        group_sums_out,
        sum,
        sum_group_size,
        element_count,
        activation_tile_index,
        batch_index,
        thread_context
    );
  }
}
