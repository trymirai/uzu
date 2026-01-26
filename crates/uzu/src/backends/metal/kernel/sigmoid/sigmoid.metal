#include <metal_stdlib>
#include "../definitions.metal"

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(Sigmoid) (
    const device T* input,
    device T* output,
    const constant uint& total_elements,
    uint tid AXIS(total_elements, 256)
) {
  if (tid >= total_elements)
    return;
  float x = float(input[tid]);
  output[tid] = T(1.0f / (1.0f + exp(-x)));
}