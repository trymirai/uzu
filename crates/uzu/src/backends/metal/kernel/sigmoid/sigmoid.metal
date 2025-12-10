#include <metal_stdlib>
#include "../definitions.metal"
using namespace metal;

// Apply sigmoid element-wise to logits
template <typename T>
kernel void apply_sigmoid(
    const device T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant int& total_elements [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
  if (tid >= total_elements)
    return;
  float x = float(input[tid]);
  output[tid] = T(1.0f / (1.0f + exp(-x)));
}

// Explicit instantiations
template [[host_name("apply_sigmoid_f16")]] [[kernel]] void apply_sigmoid<half>(
    const device half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant int& total_elements [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
);

template [[host_name("apply_sigmoid_f32")]] [[kernel]] void apply_sigmoid<
    float>(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& total_elements [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
);

template [[host_name("apply_sigmoid_bf16")]] [[kernel]] void apply_sigmoid<
    bfloat>(
    const device bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant int& total_elements [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
);
