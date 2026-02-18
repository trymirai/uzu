#include <metal_stdlib>
#include "../definitions.metal"

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(TensorAddSwap)(
    device T* skip_buffer,
    device T* main_buffer,
    constant uint& length,
    const uint position AXIS(length, 32)
) {
  T skip_value = skip_buffer[position];
  T main_value = main_buffer[position];
  T result = main_value + skip_value;

  main_buffer[position] = result;
  skip_buffer[position] = result;
}