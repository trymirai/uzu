#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

template <typename T>
void clamp_tensor(
    device const T* input,
    device T* output,
    const constant int& n,
    const constant float& min_value,
    const constant float& max_value,
    const uint tid
) {
  if ((int)tid >= n) {
    return;
  }
  const float x = float(input[tid]);
  const float y = clamp(x, min_value, max_value);
  output[tid] = (T)y;
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(AudioClamp)(
    device const T* input,
    device T* output,
    const constant int& n,
    const constant float& min_value,
    const constant float& max_value,
    uint tid AXIS(n, 256)
) {
  clamp_tensor<T>(input, output, n, min_value, max_value, tid);
}
