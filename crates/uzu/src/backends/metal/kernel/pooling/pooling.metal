#include <metal_stdlib>
#include "../definitions.metal"

enum PoolingType : uint {
  POOL_CLS = 0,
  POOL_MEAN = 1,
};

SPECIALIZE(T, float, half, bfloat) KERNEL(Pooling) (
    const device T* input,
    device T* output,
    constant int& seq_len,
    constant int& hidden_dim,
    constant int& batch_size,
    constant uint& pooling_type,
    uint dim_idx AXIS(hidden_dim, 16),
    uint batch_idx AXIS(batch_size, 16)
) {
  if (dim_idx >= hidden_dim)
    return;

  if (pooling_type == POOL_CLS) {
    output[batch_idx * hidden_dim + dim_idx] = input[batch_idx * seq_len * hidden_dim + dim_idx];
    return;
  }

  float sum = 0.0f;
  for (int seq_idx = 0; seq_idx < seq_len; seq_idx++) {
    sum += float(
        input[batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + dim_idx]
    );
  }
  output[batch_idx * hidden_dim + dim_idx] = T(sum / float(seq_len));
}