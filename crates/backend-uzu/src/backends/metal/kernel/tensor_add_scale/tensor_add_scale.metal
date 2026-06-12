#include <metal_stdlib>
#include "../common/dsl.h"

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(TensorAddScale)(
    const device T* input OPTIONAL(!in_place),
    const device T* bias,
    device T* output,
    constant uint& num_cols,
    constant uint& length,
    constant float& scale,
    const uint position AXIS(length, 32),
    const bool in_place SPECIALIZE
) {
  if (in_place) {
    input = output;
  }
  uint col = position % num_cols;
  float value = (float(input[position]) + float(bias[col])) * scale;
  output[position] = T(value);
}
