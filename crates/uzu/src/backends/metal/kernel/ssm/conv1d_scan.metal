#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

constant int activation_type [[function_constant(0)]];

constant int ACTIVATION_IDENTITY = 0;
constant int ACTIVATION_SILU = 1;
constant int ACTIVATION_GELU = 2;

constant uint CONV_SCAN_THREADS = 32u;
constant uint CONV_SCAN_TOKENS_PER_THREAD = 4u;
constant uint CONV_SCAN_BLOCK_TOKENS =
    CONV_SCAN_THREADS * CONV_SCAN_TOKENS_PER_THREAD;
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
    const size_t state_offset = channel_idx * state_stride;
    const size_t weight_offset = size_t(channel_idx) * size_t(kernel_value);
    const device T* w_row = w + weight_offset;
    const bool has_bias = b != nullptr;
    const float bias = has_bias ? float(b[channel_idx]) : 0.0f;

    if (tap_count <= 0 || tap_count > int(CONV_MAX_TAP)) {
        // Fallback to sequential scan identical to the previous kernel
        device T* state_row = state + state_offset;
        for (size_t token = 0; token < suffix_len; ++token) {
            const size_t x_idx = token * row_stride + channel_idx;
            const T input_val = x[x_idx];
            float acc = bias;
            for (int tap = 0; tap < tap_count; ++tap) {
                acc += float(w_row[tap]) * float(state_row[tap]);
            }
            acc += float(w_row[tap_count]) * float(input_val);
            y[x_idx] = apply_activation_fn(static_cast<T>(acc), activation_type);

            if (tap_count > 0) {
                for (int tap = 0; tap < tap_count - 1; ++tap) {
                    state_row[tap] = state_row[tap + 1];
                }
                state_row[tap_count - 1] = input_val;
            }
        }
        return;
    }

    threadgroup float chunk_inputs[CONV_SCAN_THREADS * CONV_SCAN_TOKENS_PER_THREAD];
    threadgroup float chunk_state_shared[CONV_SCAN_THREADS * CONV_MAX_TAP];
    threadgroup float prefix_state_shared[CONV_SCAN_THREADS * CONV_MAX_TAP];
    threadgroup ushort chunk_drop_shared[CONV_SCAN_THREADS];
    threadgroup ushort prefix_drop_shared[CONV_SCAN_THREADS];
    threadgroup float state_block_start[CONV_MAX_TAP];
    threadgroup float state_current[CONV_MAX_TAP];
    threadgroup float block_total_state[CONV_MAX_TAP];
    threadgroup ushort block_total_drop_mem[1];

    // Load the persistent conv state into threadgroup memory
    for (int tap = int(lane); tap < tap_count; tap += int(CONV_SCAN_THREADS)) {
        state_current[tap] = float(state[state_offset + tap]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (size_t block_start = 0; block_start < suffix_len; block_start += CONV_SCAN_BLOCK_TOKENS) {
        const uint block_tokens = (uint)min(
            (size_t)CONV_SCAN_BLOCK_TOKENS, suffix_len - block_start);
        if (block_tokens == 0) {
            continue;
        }
        const uint active_threads = (block_tokens + CONV_SCAN_TOKENS_PER_THREAD - 1)
            / CONV_SCAN_TOKENS_PER_THREAD;

        for (int tap = int(lane); tap < tap_count; tap += int(CONV_SCAN_THREADS)) {
            state_block_start[tap] = state_current[tap];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float chunk_inputs_local[CONV_SCAN_TOKENS_PER_THREAD];
        const uint tokens_before = lane * CONV_SCAN_TOKENS_PER_THREAD;
        uint chunk_token_count = 0u;
        if (lane < CONV_SCAN_THREADS && tokens_before < block_tokens) {
            const uint remaining = block_tokens - tokens_before;
            chunk_token_count = min(CONV_SCAN_TOKENS_PER_THREAD, remaining);
        }
        const size_t chunk_base = block_start + size_t(tokens_before);
        for (uint i = 0; i < chunk_token_count; ++i) {
            const size_t token = chunk_base + i;
            const size_t x_idx = token * row_stride + channel_idx;
            chunk_inputs_local[i] = float(x[x_idx]);
        }
        for (uint i = chunk_token_count; i < CONV_SCAN_TOKENS_PER_THREAD; ++i) {
            chunk_inputs_local[i] = 0.0f;
        }
        for (uint i = 0; i < CONV_SCAN_TOKENS_PER_THREAD; ++i) {
            chunk_inputs[lane * CONV_SCAN_TOKENS_PER_THREAD + i] = chunk_inputs_local[i];
        }

        chunk_drop_shared[lane] = (ushort)metal::min(
            (uint)chunk_token_count, (uint)tap_count);
        threadgroup float* chunk_vec = chunk_state_shared + lane * CONV_MAX_TAP;
        for (int tap = 0; tap < tap_count; ++tap) {
            chunk_vec[tap] = 0.0f;
        }
        const uint keep = metal::min((uint)chunk_token_count, (uint)tap_count);
        for (uint t = 0; t < keep; ++t) {
            const uint src_idx = chunk_token_count - keep + t;
            chunk_vec[tap_count - keep + t] = chunk_inputs_local[src_idx];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint offset = 1; offset < active_threads; offset <<= 1) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane < active_threads && lane >= offset) {
                const ushort prev_drop = chunk_drop_shared[lane - offset];
                threadgroup const float* prev_vec =
                    chunk_state_shared + (lane - offset) * CONV_MAX_TAP;
                ushort cur_drop = chunk_drop_shared[lane];
                threadgroup float* cur_vec =
                    chunk_state_shared + lane * CONV_MAX_TAP;

                const ushort combined_drop = ushort(
                    metal::min(tap_count, int(cur_drop) + int(prev_drop)));
                for (int tap = 0; tap < tap_count; ++tap) {
                    float shifted_prev = 0.0f;
                    const int idx = tap + int(cur_drop);
                    if (idx < tap_count) {
                        shifted_prev = prev_vec[idx];
                    }
                    cur_vec[tap] = shifted_prev + cur_vec[tap];
                }
                chunk_drop_shared[lane] = combined_drop;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane == 0) {
            if (active_threads > 0) {
                block_total_drop_mem[0] = chunk_drop_shared[active_threads - 1];
                threadgroup const float* block_vec =
                    chunk_state_shared + (active_threads - 1) * CONV_MAX_TAP;
                for (int tap = 0; tap < tap_count; ++tap) {
                    block_total_state[tap] = block_vec[tap];
                }
            } else {
                block_total_drop_mem[0] = 0;
                for (int tap = 0; tap < tap_count; ++tap) {
                    block_total_state[tap] = 0.0f;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < active_threads) {
            if (lane == 0) {
                prefix_drop_shared[0] = 0;
                threadgroup float* prefix_vec = prefix_state_shared;
                for (int tap = 0; tap < tap_count; ++tap) {
                    prefix_vec[tap] = 0.0f;
                }
            } else {
                prefix_drop_shared[lane] = chunk_drop_shared[lane - 1];
                threadgroup float* prefix_vec =
                    prefix_state_shared + lane * CONV_MAX_TAP;
                threadgroup const float* src_vec =
                    chunk_state_shared + (lane - 1) * CONV_MAX_TAP;
                for (int tap = 0; tap < tap_count; ++tap) {
                    prefix_vec[tap] = src_vec[tap];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane == 0) {
            const ushort total_drop = block_total_drop_mem[0];
            for (int tap = 0; tap < tap_count; ++tap) {
                float shifted = 0.0f;
                const int idx = tap + int(total_drop);
                if (idx < tap_count) {
                    shifted = state_block_start[idx];
                }
                state_current[tap] = shifted + block_total_state[tap];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < active_threads) {
            float local_state[CONV_MAX_TAP];
            const ushort drop = prefix_drop_shared[lane];
            threadgroup const float* prefix_vec =
                prefix_state_shared + lane * CONV_MAX_TAP;
            for (int tap = 0; tap < tap_count; ++tap) {
                float shifted = 0.0f;
                const int idx = tap + int(drop);
                if (idx < tap_count) {
                    shifted = state_block_start[idx];
                }
                local_state[tap] = shifted + prefix_vec[tap];
            }

            const uint tokens_remaining =
                (tokens_before < block_tokens) ? (block_tokens - tokens_before) : 0u;
            const uint tokens_to_process = min(
                CONV_SCAN_TOKENS_PER_THREAD, tokens_remaining);

            for (uint i = 0; i < tokens_to_process; ++i) {
                const size_t token = block_start + size_t(tokens_before + i);
                const size_t x_idx = token * row_stride + channel_idx;
                const float input_val = chunk_inputs_local[i];

                float acc = bias;
                for (int tap = 0; tap < tap_count; ++tap) {
                    acc += float(w_row[tap]) * local_state[tap];
                }
                acc += float(w_row[tap_count]) * input_val;

                y[x_idx] = apply_activation_fn(static_cast<T>(acc), activation_type);
                shift_register(local_state, tap_count, input_val);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int tap = int(lane); tap < tap_count; tap += int(CONV_SCAN_THREADS)) {
        state[state_offset + tap] = static_cast<T>(state_current[tap]);
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
