#include <metal_stdlib>
#include "../common/activation_quantization.h"
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../generated/activation_transform.h"
#include "../hadamard_transform/hadamard_transform.h"

using namespace metal;
using namespace uzu::activation_transform;

#define NUM_SIMDGROUPS 4
#define NUM_THREADS NUM_SIMDGROUPS* METAL_SIMD_SIZE
#define TILE_ELEMENTS NUM_THREADS* HADAMARD_VECTOR_SIZE

#define QUANTIZED (ops == ActivationTransformOp::Quantize || ops == ActivationTransformOp::QuantizeWithGroupSums)
#define EMITS_GROUP_SUMS (ops == ActivationTransformOp::QuantizeWithGroupSums)

template <typename T, typename BiasT>
VARIANTS(T, float, bfloat)
VARIANTS(BiasT, float, bfloat)
PUBLIC KERNEL(ActivationTransform)(
    const device T* input OPTIONAL(!in_place),
    device T* fp_out OPTIONAL(!QUANTIZED),
    const device BiasT* bias OPTIONAL(has_bias),
    device int8_t* q_out OPTIONAL(QUANTIZED),
    device float* scales_out OPTIONAL(QUANTIZED),
    device int32_t* group_sums_out OPTIONAL(EMITS_GROUP_SUMS),
    const device int32_t* rht_factors,
    constant uint& batch_size,
    constant uint& element_count,
    const ActivationTransformOp ops SPECIALIZE,
    const bool grouped_by_weight_nibble SPECIALIZE,
    const bool in_place SPECIALIZE,
    const uint activation_scale_group_size SPECIALIZE,
    const uint sum_group_size SPECIALIZE,
    const bool has_bias SPECIALIZE,
    uint activation_tile_index GROUPS(element_count.div_ceil(TILE_ELEMENTS)),
    uint batch_index GROUPS(batch_size),
    uint thread_index THREADS(NUM_THREADS),
    const ThreadContext thread_context
) {
  if (in_place) {
    input = reinterpret_cast<const device T*>(fp_out);
  }

  const bool input_rht = ops != ActivationTransformOp::OutputRht;
  const ushort lane_index = thread_context.simd_lane_id;
  const uint first_index = activation_tile_index * TILE_ELEMENTS + thread_index * HADAMARD_VECTOR_SIZE;
  const bool in_bounds = first_index < element_count;
  const uint element_index = batch_index * element_count + first_index;

  float4 values = 0.0f;
  int4 factors = 0;
  if (in_bounds) {
    values = float4(load_hadamard_vector(input + element_index));
    factors = load_hadamard_vector(rht_factors + first_index);
    if (input_rht) {
      values *= float4(factors);
    }
  }
  values = simdgroup_hadamard_transform_vector(lane_index, values);
  if (!input_rht) {
    values *= float4(factors);
  }

  if (!QUANTIZED) {
    if (in_bounds) {
      if (has_bias) {
        values = float4(vec<T, HADAMARD_VECTOR_SIZE>(values)) + float4(load_hadamard_vector(bias + first_index));
      }
      store_hadamard_vector(fp_out + element_index, values);
    }
    return;
  }

  store_quantized_activation_vector(
      values,
      lane_index,
      in_bounds,
      first_index,
      element_count,
      batch_index,
      activation_scale_group_size,
      grouped_by_weight_nibble,
      EMITS_GROUP_SUMS,
      sum_group_size,
      q_out,
      scales_out,
      group_sums_out
  );
}
