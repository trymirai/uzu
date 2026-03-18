#include <metal_stdlib>
#include "../common/dsl.h"

#include "activation.h"

template <typename T>
VARIANTS(T, half, float, bfloat)
PUBLIC KERNEL(Activation) (
    const device T* input OPTIONAL(!in_place),
    device T* output,
    const constant uint& n,
    const constant uint& act_type,
    const bool in_place SPECIALIZE,
    uint tid AXIS(n, 256)
) {
  if (in_place) {
    input = reinterpret_cast<const device T*>(output);
  }
  output[tid] = activate(input[tid], act_type);
}
