#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

template <typename T>
inline T silu(T x) {
    float xf = float(x);
    float y = 1.0f / (1.0f + fast::exp(-fabs(xf)));
    float out = (xf < 0.0f) ? (1.0f - y) * xf : y * xf;
    return static_cast<T>(out);
}

template <typename T>
inline T gelu(T x) {
    float xf = float(x);
    constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    float inner = SQRT_2_OVER_PI * (xf + 0.044715f * xf * xf * xf);
    float out = 0.5f * xf * (1.0f + tanh(inner));
    return static_cast<T>(out);
}

template <typename T>
kernel void ssm_activation_kernel(
    device       T* data [[ buffer(0) ]],
    constant const int& activation_type [[ buffer(1) ]],
    constant const int& row_stride [[ buffer(2) ]],
    constant const int& suffix_length [[ buffer(3) ]],
    uint2 grid_idx [[ thread_position_in_grid ]],
    uint2 grid_size [[ threads_per_grid ]]
) {
    const int row = static_cast<int>(grid_idx.x);
    const int col = static_cast<int>(grid_idx.y);

    if (row >= suffix_length || col >= row_stride) {
        return;
    }

    const int idx = row * row_stride + col;
    T value = data[idx];
    switch (activation_type) {
        case 1:
            value = silu(value);
            break;
        case 2:
            value = gelu(value);
            break;
        default:
            break;
    }
    data[idx] = value;
}

#define instantiate_ssm_activation_kernel(type_name, type)      \
  template [[host_name("ssm_activation_kernel_" #type_name)]]   \
  kernel void ssm_activation_kernel<type>(                      \
    device       type* data [[ buffer(0) ]],                    \
    constant const int& activation_type [[ buffer(1) ]],        \
    constant const int& row_stride [[ buffer(2) ]],             \
    constant const int& suffix_length [[ buffer(3) ]],          \
    uint2 grid_idx [[ thread_position_in_grid ]],               \
    uint2 grid_size [[ threads_per_grid ]]                      \
  );

instantiate_ssm_activation_kernel(float, float);
instantiate_ssm_activation_kernel(bfloat, bfloat);
instantiate_ssm_activation_kernel(half, half);

#undef instantiate_ssm_activation_kernel
