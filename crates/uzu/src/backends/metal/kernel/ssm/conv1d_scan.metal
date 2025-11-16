#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

constant int activation_type [[function_constant(0)]];

constant int ACTIVATION_IDENTITY = 0;
constant int ACTIVATION_SILU = 1;
constant int ACTIVATION_GELU = 2;

constant uint CONV_SCAN_THREADS = 32u;
constant uint CONV_MAX_TAP = 32u;

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

static inline void shift_register(thread float* state, int tap_count, float input_val) {
    if (tap_count <= 0) {
        return;
    }
    for (int tap = 0; tap < tap_count - 1; ++tap) {
        state[tap] = state[tap + 1];
    }
    state[tap_count - 1] = input_val;
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
    constant const uint& num_channels [[ buffer(9) ]],
    uint channel_block [[ threadgroup_position_in_grid ]],
    uint lane [[ thread_position_in_threadgroup ]]
) {
    const uint channel_idx = channel_block;
    if (channel_idx >= num_channels || lane >= CONV_SCAN_THREADS) {
        return;
    }

    const int kernel_value = kernel_size;
    if (kernel_value <= 0) {
        return;
    }

    const int tap_count = max(kernel_value - 1, 0);
    if (tap_count > int(CONV_MAX_TAP)) {
        return;
    }

    const size_t state_offset = channel_idx * state_stride;
    const size_t weight_offset = size_t(channel_idx) * size_t(kernel_value);
    const device T* w_row = w + weight_offset;
    const bool has_bias = b != nullptr;
    const float bias = has_bias ? float(b[channel_idx]) : 0.0f;

    threadgroup float state_shared[CONV_MAX_TAP];
    threadgroup float weight_shared[CONV_MAX_TAP + 1];
    threadgroup float reduction_shared[CONV_SCAN_THREADS];
    threadgroup float shared_input;

    for (int tap = int(lane); tap < tap_count; tap += int(CONV_SCAN_THREADS)) {
        state_shared[tap] = float(state[state_offset + tap]);
    }
    for (int tap = int(lane); tap < kernel_value; tap += int(CONV_SCAN_THREADS)) {
        weight_shared[tap] = float(w_row[tap]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (size_t token = 0; token < suffix_len; ++token) {
        const size_t x_idx = token * row_stride + channel_idx;
        if (lane == 0) {
            shared_input = float(x[x_idx]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float contrib = 0.0f;
        if (tap_count > 0 && lane < tap_count) {
            contrib = weight_shared[lane] * state_shared[lane];
        }
        float dot = threadgroup_cooperative_reduce_sum<CONV_SCAN_THREADS>(
            contrib, reduction_shared, lane);

        if (lane == 0) {
            float acc = bias + dot + weight_shared[tap_count] * shared_input;
            y[x_idx] = apply_activation_fn(static_cast<T>(acc), activation_type);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tap_count > 0) {
            if (lane < tap_count - 1) {
                state_shared[lane] = state_shared[lane + 1];
            } else if (lane == tap_count - 1) {
                state_shared[lane] = shared_input;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tap_count > 0) {
        for (int tap = int(lane); tap < tap_count; tap += int(CONV_SCAN_THREADS)) {
            state[state_offset + tap] = static_cast<T>(state_shared[tap]);
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
    constant const uint& num_channels [[ buffer(9) ]],       \
    uint channel_block [[ threadgroup_position_in_grid ]],   \
    uint lane [[ thread_position_in_threadgroup ]]           \
  );

instantiate_conv1d_scan_kernel(float, float);
instantiate_conv1d_scan_kernel(bfloat, bfloat);
instantiate_conv1d_scan_kernel(half, half);

#undef instantiate_conv1d_scan_kernel
