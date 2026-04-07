#include <metal_stdlib>
#include "../common/dsl.h"

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(ElementWiseMulStrided)(
    const device T* input_a,
    const device T* input_b,
    device T* output,
    constant uint& ple_dim,
    constant uint& stride,
    constant uint& layer_offset,
    constant uint& rows,
    const uint col AXIS(ple_dim, 64),
    const uint row AXIS(rows, 1)
) {
  uint a_idx = row * ple_dim + col;
  uint b_idx = row * stride + layer_offset + col;
  float a = float(input_a[a_idx]);
  float b = float(input_b[b_idx]);
  output[a_idx] = T(a * b);
}
