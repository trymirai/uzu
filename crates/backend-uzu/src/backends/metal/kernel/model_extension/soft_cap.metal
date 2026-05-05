#include <metal_stdlib>
#include "../common/dsl.h"

using namespace metal;

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(SoftCap)(
    const device T* input OPTIONAL(!in_place),
    device T* output,
    constant uint& length,
    constant float& cap,
    const bool in_place SPECIALIZE,
    const uint position AXIS(length, 256)
) {
  if (in_place) {
    input = reinterpret_cast<const device T*>(output);
  }
  float value = float(input[position]);
  float inv_cap = 1.0f / cap;
  output[position] = T(cap * fast::tanh(value * inv_cap));
}
