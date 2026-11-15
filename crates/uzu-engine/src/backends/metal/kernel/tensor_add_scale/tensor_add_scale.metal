#include <metal_stdlib>
#include "../common/dsl.h"

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(TensorAddScale)(
    const device T* input OPTIONAL(!in_place),
    const device T* bias OPTIONAL(has_bias),
    device T* output,
    constant uint& num_cols,
    constant uint& length,
    constant float& scale,
    const uint position AXIS(length, 32),
    const bool in_place SPECIALIZE,
    const bool has_bias SPECIALIZE
) {
  if (in_place) {
    input = output;
  }
  uint col = position % num_cols;
  float bias_value = has_bias ? float(bias[col]) : 0.0f;
  float value = (float(input[position]) + bias_value) * scale;
  output[position] = T(value);
}
