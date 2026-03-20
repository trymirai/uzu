#include <metal_stdlib>
#include "../common/dsl.h"

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(SigmoidGate)(
    const device T* gate,
    device T* output,
    const constant uint& total_elements,
    const uint tid AXIS(total_elements, 256)
) {
  if (tid >= total_elements)
    return;
  float sigmoid = 1.0f / (1.0f + exp(-float(gate[tid])));
  output[tid] = T(float(output[tid]) * sigmoid);
}
