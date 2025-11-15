#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

constant int activation_type [[function_constant(0)]];

constant int ACTIVATION_IDENTITY = 0;
constant int ACTIVATION_SILU = 1;
constant int ACTIVATION_GELU = 2;

template <typename T>
inline T apply_silu(T x) {
    float xf = float(x);
    float y = 1.0f / (1.0f + fast::exp(-fabs(xf)));
    float out = (xf < 0.0f) ? (1.0f - y) * xf : y * xf;
    return static_cast<T>(out);
}

template <typename T>
inline T apply_gelu(T x) {
    float xf = float(x);
    return static_cast<T>(0.5f * xf * (1.0f + fast::tanh(0.797885f * (xf + 0.044715f * xf * xf * xf))));
}

template <typename T>
inline T apply_activation(T x, int activation_type) {
    if (activation_type == ACTIVATION_SILU) {
        return apply_silu(x);
    } else if (activation_type == ACTIVATION_GELU) {
        return apply_gelu(x);
    } else {
        return x; // Identity
    }
}

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

        // Apply activation (fused)
        y[x_idx] = apply_activation(acc, activation_type);

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
