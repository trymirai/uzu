#include <metal_stdlib>
#include "../definitions.metal"

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(TensorAddBias)(
    const device T* input,
    const device T* bias,
    device T* output,
    constant uint& num_cols,
    constant uint& length,
    const uint position AXIS(length, 32)
) {
  uint col = position % num_cols;
  output[position] = input[position] + bias[col];
}