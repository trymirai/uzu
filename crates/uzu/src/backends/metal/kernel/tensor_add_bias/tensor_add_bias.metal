#include <metal_stdlib>
#include "../definitions.metal"

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(TensorAddBias)(
    const device T* input OPTIONAL(!in_place),
    const device T* bias,
    device T* output,
    constant uint& num_cols,
    constant uint& length,
    const uint position AXIS(length, 32),
    const bool in_place SPECIALIZE
) {
  if (in_place) {
    input = output;
  }

  uint col = position % num_cols;
  output[position] = input[position] + bias[col];
}