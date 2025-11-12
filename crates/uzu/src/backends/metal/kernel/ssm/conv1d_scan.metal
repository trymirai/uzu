#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

template <typename T>
kernel void conv1d_scan_kernel(
    device const T* x [[ buffer(0) ]],      // (suffix, channels)
    device const T* w [[ buffer(1) ]],      // (channels, kernel)
    device const T* b [[ buffer(2) ]],      // optional (channels)
    device T* state [[ buffer(3) ]],        // (channels, kernel-1)
    device T* y [[ buffer(4) ]],            // (suffix, channels)
    constant const size_t& suffix_len [[ buffer(5) ]],
    constant const int& kernel_size [[ buffer(6) ]],
    constant const size_t& row_stride [[ buffer(7) ]],
    constant const size_t& state_stride [[ buffer(8) ]],
    uint channel_idx [[ thread_position_in_grid ]],
    uint channels [[ threads_per_grid ]]
) {
    if (channel_idx >= channels) {
        return;
    }

    const int tap_count = kernel_size > 0 ? kernel_size - 1 : 0;
    device T* state_row = state + channel_idx * state_stride;
    device const T* w_row = w + channel_idx * kernel_size;
    const T bias = b ? b[channel_idx] : T(0);

    for (size_t token = 0; token < suffix_len; ++token) {
        const size_t x_idx = token * row_stride + channel_idx;
        const T input_val = x[x_idx];

        T acc = bias;
        #pragma unroll
        for (int tap = 0; tap < tap_count; ++tap) {
            acc += w_row[tap] * state_row[tap];
        }
        acc += w_row[tap_count] * input_val;
        y[x_idx] = acc;

        if (tap_count > 0) {
            for (int tap = 0; tap < tap_count - 1; ++tap) {
                state_row[tap] = state_row[tap + 1];
            }
            state_row[tap_count - 1] = input_val;
        }
    }
}

#define instantiate_conv1d_scan_kernel(type_name, type)      \
  template [[host_name("conv1d_scan_kernel_" #type_name)]]   \
  kernel void conv1d_scan_kernel<type>(                      \
    device const type* x [[ buffer(0) ]],                    \
    device const type* w [[ buffer(1) ]],                    \
    device const type* b [[ buffer(2) ]],                    \
    device type* state [[ buffer(3) ]],                      \
    device type* y [[ buffer(4) ]],                          \
    constant const size_t& suffix_len [[ buffer(5) ]],       \
    constant const int& kernel_size [[ buffer(6) ]],         \
    constant const size_t& row_stride [[ buffer(7) ]],       \
    constant const size_t& state_stride [[ buffer(8) ]],     \
    uint channel_idx [[ thread_position_in_grid ]],          \
    uint channels [[ threads_per_grid ]]                     \
  );

instantiate_conv1d_scan_kernel(float, float);
instantiate_conv1d_scan_kernel(bfloat, bfloat);
instantiate_conv1d_scan_kernel(half, half);

#undef instantiate_conv1d_scan_kernel
