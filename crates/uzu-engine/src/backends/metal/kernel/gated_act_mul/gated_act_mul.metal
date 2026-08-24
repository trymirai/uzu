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
    const bool interleaved SPECIALIZE,
    const bool use_hadamard SPECIALIZE,
    const uint activation_scale_group_size SPECIALIZE,
    const uint sum_group_size SPECIALIZE,
    const bool custom_activation_alpha SPECIALIZE,
    const bool clip_gate SPECIALIZE,
    const bool clip_value SPECIALIZE,
    threadgroup float partial_max OPTIONAL(QUANTIZED && activation_scale_group_size > METAL_SIMD_SIZE)[NUM_SIMDGROUPS],
    threadgroup int partial_sums OPTIONAL(EMITS_GROUP_SUMS && sum_group_size > METAL_SIMD_SIZE)[NUM_SIMDGROUPS],
    uint activation_tile_index GROUPS(gated_dim.div_ceil(NUM_THREADS)),
    uint batch_idx GROUPS(batch_dim),
    uint thread_index THREADS(NUM_THREADS),
    const ThreadContext thread_context
) {
  const uint gated_idx = activation_tile_index * ACTIVATION_QUANT_TILE_SIZE + thread_index;
  const uint simdgroup_offset =
      activation_tile_index * ACTIVATION_QUANT_TILE_SIZE + (thread_index / METAL_SIMD_SIZE) * METAL_SIMD_SIZE;
  const bool element_in_bounds = gated_idx < gated_dim;
  const bool simdgroup_in_bounds = simdgroup_offset + METAL_SIMD_SIZE <= gated_dim;
  T value = static_cast<T>(0);
  T gate = static_cast<T>(0);
  if (element_in_bounds) {
    if (interleaved) {
      const uint base = batch_idx * (2 * gated_dim);
      value = act_operand[base + gated_idx];
      gate = act_operand[base + gated_dim + gated_idx];
    } else {
      value = value_operand[batch_idx * value_row_stride + value_offset + gated_idx];
      gate = act_operand[batch_idx * gated_dim + gated_idx];
    }
  }
  if (clip_gate) {
    gate = static_cast<T>(clamp(float(gate), gate_clip_min, gate_clip_max));
  }
  if (clip_value) {
    value = static_cast<T>(clamp(float(value), value_clip_min, value_clip_max));
  }
  if (!QUANTIZED) {
    T result = static_cast<T>(0);
    if (element_in_bounds && (!use_hadamard || simdgroup_in_bounds)) {
      float transformed;
      if (custom_activation_alpha && act_type == ActivationType::SILU) {
        const T activated = activate_silu_alpha(gate, activation_alpha);
        const T gated = value * activated;
        transformed = static_cast<float>(gated);
      } else {
        transformed = gated_act_mul(value, gate, act_type);
      }
      if (use_hadamard) {
        transformed = simdgroup_input_random_hadamard_transform(
            static_cast<ushort>(gated_idx % METAL_SIMD_SIZE),
            transformed,
            hadamard_factors[gated_idx]
        );
      }
      result = static_cast<T>(transformed);
    }
    if (element_in_bounds) {
      fp_out[batch_idx * gated_dim + gated_idx] = result;
    }
    return;
  }

  float result = 0.0f;
  if (simdgroup_in_bounds) {
    if (custom_activation_alpha && act_type == ActivationType::SILU) {
      const T activated = activate_silu_alpha(gate, activation_alpha);
      const T gated = value * activated;
      result = static_cast<float>(gated);
    } else {
      result = gated_act_mul(value, gate, act_type);
    }
    result = simdgroup_input_random_hadamard_transform(
        static_cast<ushort>(gated_idx % METAL_SIMD_SIZE),
        result,
        hadamard_factors[gated_idx]
    );
  }

  const float maximum = reduce_activation_quantization_group(
      fabs(result),
      activation_scale_group_size,
      partial_max,
      thread_context,
      [](float x) { return simd_max(x); },
      [](float x, float y) { return max(x, y); }
  );
  const float scale = isfinite(maximum) && maximum > 0.0f ? maximum / ACTIVATION_QUANT_INT8_MAX : 1.0f;
  const int8_t code =
      static_cast<int8_t>(clamp(round(result / scale), -ACTIVATION_QUANT_INT8_MAX, ACTIVATION_QUANT_INT8_MAX));
  if (element_in_bounds) {
    q_out[batch_idx * gated_dim + gated_idx] = code;
  }

  write_activation_quantization_group(
      scales_out,
      scale,
      activation_scale_group_size,
      gated_dim,
      activation_tile_index,
      batch_idx,
      thread_context
  );

  if (EMITS_GROUP_SUMS) {
    const int sum = reduce_activation_quantization_group(
        static_cast<int>(code),
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
        gated_dim,
        activation_tile_index,
        batch_idx,
        thread_context
    );
  }
}
