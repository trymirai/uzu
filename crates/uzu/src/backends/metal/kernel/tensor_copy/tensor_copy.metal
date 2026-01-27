#include <metal_stdlib>
#include "../definitions.metal"

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(TensorCopy)(
    const device T* src_buffer,
    device T* dst_buffer,
    constant uint& length,
    const uint position AXIS(length, 32)
) {
  dst_buffer[position] = src_buffer[position];
}