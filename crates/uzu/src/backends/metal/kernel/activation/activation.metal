#include <metal_stdlib>
#include "../definitions.metal"

#include "activation.h"

template <typename T>
VARIANTS(T, half, float, bfloat)
KERNEL(Activation) (
    const device T* input,
    device T* output,
    const constant uint& n,
    const constant uint& act_type,
    uint tid AXIS(n, 256)
) {
  output[tid] = activate(input[tid], act_type);
}
