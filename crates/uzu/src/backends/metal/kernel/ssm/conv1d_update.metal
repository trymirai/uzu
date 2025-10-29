#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

template <typename T>
inline void conv1d_update_kernel(
    device const T* x, // b, d
    device const T* w, // d, k
    device const T* b, // d
    device const T* state, // b, d, k-1
    device T* y,
    device T* next_state,
    constant const int& kernel_size,
    constant const size_t* x_strides,
    constant const size_t* state_strides,
    uint3 grid_idx,
    uint3 /*grid_size*/
) {
    const int batch_idx = grid_idx.x;
    const int channel_idx = grid_idx.y;
    const int x_idx = static_cast<int>(batch_idx * x_strides[0] + channel_idx);
    const int w_start_idx = channel_idx * kernel_size;
    const int state_start_idx = static_cast<int>(batch_idx * state_strides[0] + channel_idx * state_strides[1]);

    T temp = T(0);
    #pragma unroll
    for (int i = 0; i < kernel_size - 1; ++i) {
        temp = temp + w[w_start_idx + i] * state[state_start_idx + i];
    }

    temp = temp + w[w_start_idx + kernel_size - 1] * x[x_idx];
    temp = temp + b[channel_idx];
    y[x_idx] = temp;

    #pragma unroll
    for (int i = 0; i < kernel_size - 2; ++i) {
        next_state[state_start_idx + i] = state[state_start_idx + i + 1];
    }
    next_state[state_start_idx + kernel_size - 2] = x[x_idx];
}

#define outerArguments(T)                                                           \
(device const T* x [[ buffer(0) ]],                                                 \
 device const T* w [[ buffer(1) ]],                                                 \
 device const T* b [[ buffer(2) ]],                                                 \
 device const T* state [[ buffer(3) ]],                                            \
 device       T* y [[ buffer(4) ]],                                                 \
 device       T* next_state [[ buffer(5) ]],                                       \
 constant const int& kernel_size [[ buffer(6) ]],                                   \
 constant const size_t* x_strides [[ buffer(7) ]],                                   \
 constant const size_t* state_strides [[ buffer(8) ]],                               \
 uint3 grid_idx [[ thread_position_in_grid ]],                                      \
 uint3 grid_size [[ threads_per_grid ]])

#define innerArguments (x, w, b, state, y, next_state, kernel_size, x_strides, state_strides, grid_idx, grid_size)

generateKernels(conv1d_update_kernel)

#undef outerArguments
#undef innerArguments


