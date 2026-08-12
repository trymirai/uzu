#pragma once

#include <metal_stdlib>

#include "../activation/activations.h"
#include "../hadamard_transform/hadamard_transform.h"

using namespace metal;
using namespace uzu::activation_type;

template <typename T>
static METAL_FUNC T gated_act_mul_result(
    T value,
    T gate,
    ActivationType act_type,
    bool use_hadamard,
    uint index,
    const device int32_t* factors
) {
  T result = value * activate(gate, act_type);

  if (use_hadamard) {
    result =
        simdgroup_input_random_hadamard_transform(static_cast<ushort>(index % METAL_SIMD_SIZE), result, factors[index]);
  }

  return result;
}
