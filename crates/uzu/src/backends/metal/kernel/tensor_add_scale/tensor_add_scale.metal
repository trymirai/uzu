#include <metal_stdlib>
#include "../definitions.metal"

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(TensorAddScale)(
    const device T* input,
    const device T* bias,
    device T* output,
    constant uint& num_cols,
    constant uint& length,
    constant float& scale,
    const uint position AXIS(length, 32)
) {
  uint col = position % num_cols;
  float value = (float(input[position]) + float(bias[col])) * scale;
  output[position] = T(value);
}
