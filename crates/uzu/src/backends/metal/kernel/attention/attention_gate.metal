#include <metal_stdlib>
#include "../definitions.metal"

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(AttentionGate)(
    const device T* qkv,
    device T* output,
    const constant uint& num_heads,
    const constant uint& num_groups,
    const constant uint& head_dim,
    const constant uint& total_heads,
    const constant uint& suffix_length,
    const uint token_index AXIS(suffix_length, 1),
    const uint head_index AXIS(num_heads, 1),
    const uint dim_index AXIS(head_dim, 64)
) {
  const uint qkv_stride = total_heads * head_dim;
  const uint gate_offset = (num_heads + 2 * num_groups) * head_dim;

  const uint gate_idx = token_index * qkv_stride + gate_offset +
                        head_index * head_dim + dim_index;
  const uint output_idx =
      (token_index * num_heads + head_index) * head_dim + dim_index;

  float sigmoid = 1.0f / (1.0f + exp(-float(qkv[gate_idx])));
  output[output_idx] = T(float(output[output_idx]) * sigmoid);
}
