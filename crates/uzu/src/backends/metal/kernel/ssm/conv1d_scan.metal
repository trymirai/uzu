#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

constant int activation_type [[function_constant(0)]];

constant int ACTIVATION_IDENTITY = 0;
constant int ACTIVATION_SILU = 1;
constant int ACTIVATION_GELU = 2;

constant uint CONV_SCAN_THREADS = 32u;

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
inline T apply_activation_fn(T x, int activation_type) {
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
    device const T* state_in [[ buffer(3) ]],// (channels, kernel-1)
    device T* y [[ buffer(4) ]],            // (suffix, channels)
    device T* state_out [[ buffer(5) ]],    // (channels, kernel-1)
    constant const size_t& suffix_len [[ buffer(6) ]],
    constant const int& kernel_size [[ buffer(7) ]],
    constant const size_t& row_stride [[ buffer(8) ]],
    constant const size_t& state_stride [[ buffer(9) ]],
    constant const uint& num_channels [[ buffer(10) ]],
    uint3 grid_idx [[ thread_position_in_grid ]]
) {
    const int kernel_value = kernel_size;
    if (kernel_value <= 0) {
        return;
    }

    const int tap_count = max(kernel_value - 1, 0);
    const size_t work_len = suffix_len + size_t(tap_count);

    const uint token_idx = grid_idx.x;
    const uint channel_idx = grid_idx.y;
    if (channel_idx >= num_channels || token_idx >= work_len) {
        return;
    }

    const size_t state_offset = size_t(channel_idx) * state_stride;
    const size_t weight_offset = size_t(channel_idx) * size_t(kernel_value);
    const device T* w_row = w + weight_offset;
    const bool has_bias = b != nullptr;

    if (token_idx < suffix_len) {
        float acc = has_bias ? float(b[channel_idx]) : 0.0f;
        for (int tap = 0; tap < kernel_value; ++tap) {
            const size_t padded_idx = size_t(token_idx) + size_t(tap);
            float sample = 0.0f;
            if (padded_idx < size_t(tap_count)) {
                sample = float(state_in[state_offset + padded_idx]);
            } else {
                const size_t x_idx = (padded_idx - size_t(tap_count)) * row_stride + channel_idx;
                sample = float(x[x_idx]);
            }
            acc += float(w_row[tap]) * sample;
        }

        const size_t y_idx = size_t(token_idx) * row_stride + channel_idx;
        y[y_idx] = apply_activation_fn(static_cast<T>(acc), activation_type);
    } else if (tap_count > 0) {
        const size_t tap = size_t(token_idx - suffix_len);
        if (tap >= size_t(tap_count)) {
            return;
        }
        const size_t padded_idx = suffix_len + tap;
        float sample = 0.0f;
        if (padded_idx < size_t(tap_count)) {
            sample = float(state_in[state_offset + padded_idx]);
        } else {
            const size_t x_idx = (padded_idx - size_t(tap_count)) * row_stride + channel_idx;
            sample = float(x[x_idx]);
        }
        state_out[state_offset + tap] = static_cast<T>(sample);
    }
}

#define instantiate_conv1d_scan_kernel(type_name, type)       \
  template [[host_name("conv1d_scan_kernel_" #type_name)]]    \
  kernel void conv1d_scan_kernel<type>(                       \
    device const type* x [[ buffer(0) ]],                     \
    device const type* w [[ buffer(1) ]],                     \
    device const type* b [[ buffer(2) ]],                     \
    device const type* state_in [[ buffer(3) ]],              \
    device type* y [[ buffer(4) ]],                           \
    device type* state_out [[ buffer(5) ]],                   \
    constant const size_t& suffix_len [[ buffer(6) ]],        \
    constant const int& kernel_size [[ buffer(7) ]],          \
    constant const size_t& row_stride [[ buffer(8) ]],        \
    constant const size_t& state_stride [[ buffer(9) ]],      \
    constant const uint& num_channels [[ buffer(10) ]],       \
    uint3 grid_idx [[ thread_position_in_grid ]]              \
  );

instantiate_conv1d_scan_kernel(float, float);
instantiate_conv1d_scan_kernel(bfloat, bfloat);
instantiate_conv1d_scan_kernel(half, half);

#undef instantiate_conv1d_scan_kernel
