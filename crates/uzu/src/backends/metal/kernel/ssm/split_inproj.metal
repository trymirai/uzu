#include <metal_stdlib>
#include "../definitions.metal"

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(SplitInProj)(
    device const T* input,
    device T* conv_out,
    device T* z_out,
    device T* dt_out,
    device const T* z_bias,
    constant const uint& suffix_length,
    constant const uint& total_dim,
    constant const uint& conv_dim,
    constant const uint& inner_dim,
    constant const uint& num_heads,
    const uint row AXIS(suffix_length, 16),
    const uint col AXIS(total_dim, 16)
) {
  const int input_idx = row * total_dim + col;
  if (col < conv_dim) {
    const int dst = row * conv_dim + col;
    conv_out[dst] = input[input_idx];
  } else if (col < conv_dim + inner_dim) {
    const int dst = row * inner_dim + (col - conv_dim);
    const int bias_idx = col - conv_dim;
    z_out[dst] = input[input_idx] + z_bias[bias_idx];
  } else if (col < conv_dim + inner_dim + num_heads) {
    const int dst = row * num_heads + (col - conv_dim - inner_dim);
    dt_out[dst] = input[input_idx];
  }
}