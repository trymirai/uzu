#include <metal_stdlib>
#include "../definitions.metal"

// CLS pooling: Extract first token [batch, seq_len, hidden_dim] → [batch,
// hidden_dim]
SPECIALIZE(T, float, half, bfloat) KERNEL(PoolingCls) (
    const device T* input,
    device T* output,
    constant uint& seq_len,
    constant uint& hidden_dim,
    constant uint& batch_size,
    uint dim_idx AXIS(hidden_dim, 16),
    uint batch_idx AXIS(batch_size, 16)
) {
  output[batch_idx * hidden_dim + dim_idx] =
      input[batch_idx * seq_len * hidden_dim + dim_idx];
}

// Mean pooling: Average across sequence [batch, seq_len, hidden_dim] → [batch,
// hidden_dim]
SPECIALIZE(T, float, half, bfloat) KERNEL(PoolingMean) (
    const device T* input,
    device T* output,
    constant uint& seq_len,
    constant uint& hidden_dim,
    constant uint& batch_size,
    uint dim_idx AXIS(hidden_dim, 16),
    uint batch_idx AXIS(batch_size, 16)
) {
  float sum = 0.0f;
  for (uint seq_idx = 0; seq_idx < seq_len; seq_idx++) {
    sum += float(
        input[batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + dim_idx]
    );
  }
  output[batch_idx * hidden_dim + dim_idx] = T(sum / float(seq_len));
}
