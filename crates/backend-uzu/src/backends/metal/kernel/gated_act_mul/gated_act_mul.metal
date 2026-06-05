#include <metal_stdlib>
#include "../activation/activations.h"
#include "../common/dsl.h"
#include "../hadamard_transform/hadamard_transform.h"

using namespace uzu::activation_type;

template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(GatedActMul) (
    const device T* act_operand,
    const device T* value_operand OPTIONAL(!interleaved),
    device T* output,
    const device int32_t* hadamard_factors OPTIONAL(use_hadamard),
    const constant uint& gated_dim,
    const constant uint& batch_dim,
    const constant uint& value_offset,
    const constant uint& value_row_stride,
    const constant ActivationType& act_type,
    const bool interleaved SPECIALIZE,
    const bool use_hadamard SPECIALIZE,
    uint gated_idx AXIS(gated_dim, 64),
    uint batch_idx AXIS(batch_dim, 1)
) {
  T activated;
  T value;
  if (interleaved) {
    uint base = batch_idx * (2 * gated_dim);
    value = act_operand[base + gated_idx];
    activated = activate(act_operand[base + gated_dim + gated_idx], act_type);
  } else {
    activated =
        activate(act_operand[batch_idx * gated_dim + gated_idx], act_type);
    value =
        value_operand[batch_idx * value_row_stride + value_offset + gated_idx];
  }
  T result = value * activated;

  if (use_hadamard) {
    result = simdgroup_input_random_hadamard_transform(
        static_cast<ushort>(gated_idx % METAL_SIMD_SIZE),
        result,
        hadamard_factors[gated_idx]
    );
  }

  output[batch_idx * gated_dim + gated_idx] = result;
}
