#include <metal_stdlib>
#include "../activation/activations.h"
#include "../common/dsl.h"
#include "../hadamard_transform/hadamard_transform.h"

using namespace uzu::activation_type;

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(GatedActMul) (
    const device T* act_operand,
    const device T* value_operand OPTIONAL(!interleaved),
    device T* output,
    const device int32_t* hadamard_factors OPTIONAL(use_hadamard),
    const constant int& inner_dim,
    const constant int& outer_dim,
    const constant int& value_offset,
    const constant int& value_row_stride,
    const constant ActivationType& act_type,
    const bool interleaved SPECIALIZE,
    const bool use_hadamard SPECIALIZE,
    uint col AXIS(inner_dim, 64),
    uint row AXIS(outer_dim, 1)
) {
  T activated;
  T value;
  if (interleaved) {
    int base = row * (2 * inner_dim);
    value = act_operand[base + col];
    activated = activate(act_operand[base + inner_dim + col], act_type);
  } else {
    activated = activate(act_operand[row * inner_dim + col], act_type);
    value = value_operand[row * value_row_stride + value_offset + col];
  }
  T result = T(float(value) * float(activated));

  if (use_hadamard) {
    result = simdgroup_input_random_hadamard_transform(
        static_cast<ushort>(col % METAL_SIMD_SIZE),
        result,
        hadamard_factors[col]
    );
  }

  output[row * inner_dim + col] = result;
}
