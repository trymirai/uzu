#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

template <typename T>
void add_tensors(
    device const T* a,
    device const T* b,
    device T* out,
    const constant int& n,
    const uint tid
) {
  if ((int)tid >= n) {
    return;
  }
  out[tid] = a[tid] + b[tid];
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(AudioAdd)(
    device const T* a,
    device const T* b,
    device T* out,
    const constant int& n,
    uint tid AXIS(n, 256)
) {
  add_tensors<T>(a, b, out, n, tid);
}
