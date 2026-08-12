#include <metal_stdlib>
#include "../common/activation_quantization.h"
#include "../common/dsl.h"
#include "../common/gated_act_mul.h"
#include "../common/thread_context.h"
#include "../generated/gated_act_mul.h"

using namespace metal;
using namespace uzu::activation_type;
using namespace uzu::gated_act_mul;

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
    const GatedActMulOp ops SPECIALIZE,
    const bool interleaved SPECIALIZE,
    const bool use_hadamard SPECIALIZE,
    const uint activation_scale_group_size SPECIALIZE,
    const uint sum_group_size SPECIALIZE,
    threadgroup float partial_max OPTIONAL(QUANTIZED && activation_scale_group_size > METAL_SIMD_SIZE)[4],
    threadgroup int partial_sums OPTIONAL(EMITS_GROUP_SUMS && sum_group_size > METAL_SIMD_SIZE)[4],
    uint activation_tile_index GROUPS(gated_dim.div_ceil(ACTIVATION_QUANT_TILE_SIZE)),
    uint batch_idx GROUPS(batch_dim),
    uint thread_index THREADS(ACTIVATION_QUANT_TILE_SIZE),
    const ThreadContext thread_context
) {
  const uint gated_idx = activation_tile_index * ACTIVATION_QUANT_TILE_SIZE + thread_index;
  const bool in_bounds = gated_idx < gated_dim;
  T value;
  T gate;
  if (in_bounds) {
    if (interleaved) {
      const uint base = batch_idx * (2 * gated_dim);
      value = act_operand[base + gated_idx];
      gate = act_operand[base + gated_dim + gated_idx];
    } else {
      value = value_operand[batch_idx * value_row_stride + value_offset + gated_idx];
      gate = act_operand[batch_idx * gated_dim + gated_idx];
    }
  }

  const T result = in_bounds ? gated_act_mul_result(value, gate, act_type, use_hadamard, gated_idx, hadamard_factors)
                             : static_cast<T>(0);
  if (!QUANTIZED) {
    if (in_bounds) {
      fp_out[batch_idx * gated_dim + gated_idx] = result;
    }
    return;
  }

  const float maximum = reduce_activation_quantization_group(
      fabs(static_cast<float>(result)),
      activation_scale_group_size,
      partial_max,
      thread_context,
      [](float x) { return simd_max(x); },
      [](float x, float y) { return max(x, y); }
  );
  const float scale = isfinite(maximum) && maximum > 0.0f ? maximum / ACTIVATION_QUANT_INT8_MAX : 1.0f;
  const int8_t code = static_cast<int8_t>(
      clamp(round(static_cast<float>(result) / scale), -ACTIVATION_QUANT_INT8_MAX, ACTIVATION_QUANT_INT8_MAX)
  );
  if (in_bounds) {
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
