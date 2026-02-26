#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

template <typename T>
void scale_tensor(
    device const T* input,
    device T* output,
    const constant int& n,
    const constant float& scale,
    const uint tid
) {
  if ((int)tid >= n) {
    return;
  }
  const float x = float(input[tid]);
  output[tid] = (T)(x * scale);
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(AudioScale)(
    device const T* input,
    device T* output,
    const constant int& n,
    const constant float& scale,
    uint tid AXIS(n, 256)
) {
  scale_tensor<T>(input, output, n, scale, tid);
}
