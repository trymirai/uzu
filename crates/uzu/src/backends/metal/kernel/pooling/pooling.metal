#include <metal_stdlib>
#include "../definitions.metal"
using namespace metal;

// CLS pooling: Extract first token [batch, seq_len, hidden_dim] → [batch,
// hidden_dim]
template <typename T>
kernel void pool_cls(
    const device T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant int& seq_len [[buffer(2)]],
    constant int& hidden_dim [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) // x: hidden_dim, y: batch
{
  int batch_idx = tid.y;
  int dim_idx = tid.x;
  if (dim_idx >= hidden_dim)
    return;

  output[batch_idx * hidden_dim + dim_idx] =
      input[batch_idx * seq_len * hidden_dim + dim_idx];
}

// Mean pooling: Average across sequence [batch, seq_len, hidden_dim] → [batch,
// hidden_dim]
template <typename T>
kernel void pool_mean(
    const device T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant int& seq_len [[buffer(2)]],
    constant int& hidden_dim [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
  int batch_idx = tid.y;
  int dim_idx = tid.x;
  if (dim_idx >= hidden_dim)
    return;

  float sum = 0.0f;
  for (int seq_idx = 0; seq_idx < seq_len; seq_idx++) {
    sum += float(
        input[batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + dim_idx]
    );
  }
  output[batch_idx * hidden_dim + dim_idx] = T(sum / float(seq_len));
}

// Explicit instantiations for f16, f32, bf16
template [[host_name("pool_cls_f16")]] [[kernel]] void pool_cls<half>(
    const device half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant int& seq_len [[buffer(2)]],
    constant int& hidden_dim [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
);

template [[host_name("pool_cls_f32")]] [[kernel]] void pool_cls<float>(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& seq_len [[buffer(2)]],
    constant int& hidden_dim [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
);

template [[host_name("pool_cls_bf16")]] [[kernel]] void pool_cls<bfloat>(
    const device bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant int& seq_len [[buffer(2)]],
    constant int& hidden_dim [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
);

template [[host_name("pool_mean_f16")]] [[kernel]] void pool_mean<half>(
    const device half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant int& seq_len [[buffer(2)]],
    constant int& hidden_dim [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
);

template [[host_name("pool_mean_f32")]] [[kernel]] void pool_mean<float>(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& seq_len [[buffer(2)]],
    constant int& hidden_dim [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
);

template [[host_name("pool_mean_bf16")]] [[kernel]] void pool_mean<bfloat>(
    const device bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant int& seq_len [[buffer(2)]],
    constant int& hidden_dim [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
);
