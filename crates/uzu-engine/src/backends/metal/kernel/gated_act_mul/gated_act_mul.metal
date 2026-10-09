#include <metal_stdlib>
#include "../common/activation_quantization.h"
#include "../common/dsl.h"
#include "../common/gated_act_mul.h"
#include "../common/thread_context.h"
#include "../generated/gated_act_mul.h"
#include "../hadamard_transform/hadamard_transform.h"

using namespace metal;
using namespace uzu::activation_type;
using namespace uzu::gated_act_mul;

#define NUM_SIMDGROUPS 4
#define NUM_THREADS NUM_SIMDGROUPS* METAL_SIMD_SIZE
#define TILE_ELEMENTS NUM_THREADS* HADAMARD_VECTOR_SIZE

#define QUANTIZED (ops == GatedActMulOp::Quantize || ops == GatedActMulOp::QuantizeWithGroupSums)
#define EMITS_GROUP_SUMS (ops == GatedActMulOp::QuantizeWithGroupSums)

template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(GatedActMul) (
    const device T* act_operand,
    const device T* value_operand OPTIONAL(!interleaved),
    device T* fp_out OPTIONAL(!QUANTIZED),
    device int8_t* q_out OPTIONAL(QUANTIZED),
    device float* scales_out OPTIONAL(QUANTIZED),
    device int32_t* group_sums_out OPTIONAL(EMITS_GROUP_SUMS),
    const device int32_t* hadamard_factors OPTIONAL(use_hadamard),
    const constant uint& gated_dim,
    const constant uint& batch_dim,
    const constant uint& value_offset,
    const constant uint& value_row_stride,
    const constant ActivationType& act_type,
    const constant float& activation_alpha OPTIONAL(custom_activation_alpha),
    const constant float& gate_clip_min OPTIONAL(clip_gate),
    const constant float& gate_clip_max OPTIONAL(clip_gate),
    const constant float& value_clip_min OPTIONAL(clip_value),
    const constant float& value_clip_max OPTIONAL(clip_value),
    const GatedActMulOp ops SPECIALIZE,
    const bool grouped_by_weight_nibble SPECIALIZE,
    const bool interleaved SPECIALIZE,
    const bool use_hadamard SPECIALIZE,
    const uint activation_scale_group_size SPECIALIZE,
    const uint sum_group_size SPECIALIZE,
    const bool custom_activation_alpha SPECIALIZE,
    const bool clip_gate SPECIALIZE,
    const bool clip_value SPECIALIZE,
    uint activation_tile_index GROUPS(gated_dim.div_ceil(TILE_ELEMENTS)),
    uint batch_idx GROUPS(batch_dim),
    uint thread_index THREADS(NUM_THREADS),
    const ThreadContext thread_context
) {
  const uint first_index = activation_tile_index * TILE_ELEMENTS + thread_index * HADAMARD_VECTOR_SIZE;
  const ushort lane_index = thread_context.simd_lane_id;
  const uint valid_count =
      first_index < gated_dim ? min(static_cast<uint>(HADAMARD_VECTOR_SIZE), gated_dim - first_index) : 0u;
  const bool use_vector_io = valid_count == HADAMARD_VECTOR_SIZE;

  vec<T, HADAMARD_VECTOR_SIZE> value = static_cast<T>(0);
  vec<T, HADAMARD_VECTOR_SIZE> gate = static_cast<T>(0);
  {
    const device T* value_row;
    const device T* gate_row;
    if (interleaved) {
      value_row = act_operand + batch_idx * (2 * gated_dim);
      gate_row = value_row + gated_dim;
    } else {
      value_row = value_operand + batch_idx * value_row_stride + value_offset;
      gate_row = act_operand + batch_idx * gated_dim;
    }
    if (use_vector_io) {
      value = load_hadamard_vector(value_row + first_index);
      gate = load_hadamard_vector(gate_row + first_index);
    } else {
      for (uint index = 0; index < valid_count; ++index) {
        value[index] = value_row[first_index + index];
        gate[index] = gate_row[first_index + index];
      }
    }
  }
  if (clip_gate) {
    gate = vec<T, HADAMARD_VECTOR_SIZE>(clamp(float4(gate), gate_clip_min, gate_clip_max));
  }
  if (clip_value) {
    value = vec<T, HADAMARD_VECTOR_SIZE>(clamp(float4(value), value_clip_min, value_clip_max));
  }

  const bool hadamard = QUANTIZED || use_hadamard;
  const bool block_in_bounds =
      (first_index / HADAMARD_TRANSFORM_BLOCK_SIZE + 1) * HADAMARD_TRANSFORM_BLOCK_SIZE <= gated_dim;
  float4 results = 0.0f;
  METAL_PRAGMA_UNROLL
  for (uint index = 0; index < HADAMARD_VECTOR_SIZE; ++index) {
    if (index < valid_count && (!hadamard || block_in_bounds)) {
      if (custom_activation_alpha && act_type == ActivationType::SILU) {
        const T activated = activate_silu_alpha(gate[index], activation_alpha);
        const T gated = value[index] * activated;
        results[index] = static_cast<float>(gated);
      } else {
        results[index] = gated_act_mul(value[index], gate[index], act_type);
      }
    }
  }
  if (hadamard) {
    if (block_in_bounds) {
      results *= float4(load_hadamard_vector(hadamard_factors + first_index));
    }
    results = simdgroup_hadamard_transform_vector(lane_index, results);
  }

  if (!QUANTIZED) {
    device T* out_row = fp_out + batch_idx * gated_dim;
    if (use_vector_io) {
      store_hadamard_vector(out_row + first_index, results);
    } else {
      for (uint index = 0; index < valid_count; ++index) {
        out_row[first_index + index] = static_cast<T>(results[index]);
      }
    }
    return;
  }

  store_quantized_activation_vector(
      results,
      lane_index,
      valid_count > 0,
      first_index,
      gated_dim,
      batch_idx,
      activation_scale_group_size,
      grouped_by_weight_nibble,
      EMITS_GROUP_SUMS,
      sum_group_size,
      q_out,
      scales_out,
      group_sums_out
  );
}
