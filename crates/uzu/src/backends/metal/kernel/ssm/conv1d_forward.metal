#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

template <typename T>
inline void conv1d_forward_kernel(
    device const T* x,      // (b, d, l)
    device const T* w,      // (d, k)
    device const T* b,      // (d)
    device T* y,            // (b, d, l)
    constant const size_t* x_strides,
    constant const int& kernel_size,
    uint3 grid_idx,
    uint3 grid_size
) {
    const int length_size = grid_size.z;
    const int batch_idx = grid_idx.x;
    const int channel_idx = grid_idx.y;
    const int length_idx = grid_idx.z;

    const int y_idx = static_cast<int>(batch_idx * x_strides[0] + channel_idx * x_strides[1] + length_idx);
    const int w_start_idx = channel_idx * kernel_size;

    T acc = T(0);

    if (length_idx + kernel_size - 1 < length_size) {
        #pragma unroll
        for (int i = 0; i < kernel_size; ++i) {
            acc = acc + w[w_start_idx + i] * x[y_idx + i];
        }
        acc = acc + b[channel_idx];
    }
    y[y_idx] = acc;
}

#define outerArguments(T)                                                      \
(device const T* x           [[ buffer(0) ]],                                   \
 device const T* w           [[ buffer(1) ]],                                   \
 device const T* b           [[ buffer(2) ]],                                   \
 device       T* y           [[ buffer(3) ]],                                   \
 constant const size_t* x_strides [[ buffer(4) ]],                              \
 constant const int& kernel_size  [[ buffer(5) ]],                              \
 uint3 grid_idx             [[ thread_position_in_grid ]],                      \
 uint3 grid_size            [[ threads_per_grid ]])

#define innerArguments (x, w, b, y, x_strides, kernel_size, grid_idx, grid_size)

generateKernels(conv1d_forward_kernel)

#undef outerArguments
#undef innerArguments


