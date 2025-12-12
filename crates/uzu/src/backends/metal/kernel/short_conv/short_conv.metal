#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

template <typename T>
kernel void short_conv_prefill_kernel(
    device const T* in_proj [[buffer(0)]],
    device const T* w [[buffer(1)]],
    device const T* b [[buffer(2)]],
    device const T* state_in [[buffer(3)]],
    device T* out [[buffer(4)]],
    device T* state_out [[buffer(5)]],
    constant const size_t& suffix_len [[buffer(6)]],
    constant const int& kernel_size [[buffer(7)]],
    constant const size_t& in_proj_stride [[buffer(8)]],
    constant const size_t& state_stride [[buffer(9)]],
    constant const uint& model_dim [[buffer(10)]],
    uint2 grid_idx [[thread_position_in_grid]]
) {
    const uint token_idx = grid_idx.x;
    const uint channel_idx = grid_idx.y;

    if (channel_idx >= model_dim || token_idx >= suffix_len) {
        return;
    }

    const int tap_count = max(kernel_size - 1, 0);
    const size_t state_offset = size_t(channel_idx) * state_stride;
    const device T* w_row = w + size_t(channel_idx) * size_t(kernel_size);
    const bool has_bias = b != nullptr;

    size_t current_in_proj_idx = size_t(token_idx) * in_proj_stride + size_t(channel_idx);
    float pre_conv_gate = float(in_proj[current_in_proj_idx]);
    float post_conv_gate = float(in_proj[current_in_proj_idx + model_dim]);
    float x_current = float(in_proj[current_in_proj_idx + 2 * model_dim]);

    float gated_input = x_current * pre_conv_gate;

    float acc = has_bias ? float(b[channel_idx]) : 0.0f;

    for (int tap = 0; tap < kernel_size; ++tap) {
        int src_token = int(token_idx) - (kernel_size - 1 - tap);
        float sample = 0.0f;

        if (src_token < 0) {
            int state_idx = tap_count + src_token;
            if (state_idx >= 0 && state_idx < tap_count) {
                sample = float(state_in[state_offset + size_t(state_idx)]);
            }
        } else if (src_token == int(token_idx)) {
            sample = gated_input;
        } else {
            size_t src_in_proj_idx = size_t(src_token) * in_proj_stride + size_t(channel_idx);
            float src_pre_gate = float(in_proj[src_in_proj_idx]);
            float src_x = float(in_proj[src_in_proj_idx + 2 * model_dim]);
            sample = src_x * src_pre_gate;
        }

        acc += float(w_row[tap]) * sample;
    }

    float gated_output = acc * post_conv_gate;

    size_t out_idx = size_t(token_idx) * model_dim + size_t(channel_idx);
    out[out_idx] = static_cast<T>(gated_output);

    if (tap_count > 0 && token_idx == 0) {
        for (int tap = 0; tap < tap_count; ++tap) {
            int src_token = int(suffix_len) - tap_count + tap;
            float sample = 0.0f;

            if (src_token < 0) {
                int old_state_idx = tap_count + src_token;
                if (old_state_idx >= 0 && old_state_idx < tap_count) {
                    sample = float(state_in[state_offset + size_t(old_state_idx)]);
                }
            } else {
                size_t src_in_proj_idx = size_t(src_token) * in_proj_stride + size_t(channel_idx);
                float src_pre_gate = float(in_proj[src_in_proj_idx]);
                float src_x = float(in_proj[src_in_proj_idx + 2 * model_dim]);
                sample = src_x * src_pre_gate;
            }

            state_out[state_offset + size_t(tap)] = static_cast<T>(sample);
        }
    }
}

template <typename T>
kernel void short_conv_decode_kernel(
    device const T* in_proj [[buffer(0)]],
    device const T* w [[buffer(1)]],
    device const T* b [[buffer(2)]],
    device const T* state [[buffer(3)]],
    device T* out [[buffer(4)]],
    device T* next_state [[buffer(5)]],
    constant const size_t& suffix_len [[buffer(6)]],
    constant const int& kernel_size [[buffer(7)]],
    constant const size_t& in_proj_stride [[buffer(8)]],
    constant const size_t& state_stride [[buffer(9)]],
    constant const uint& model_dim [[buffer(10)]],
    uint2 grid_idx [[thread_position_in_grid]]
) {
    const uint token_idx = grid_idx.x;
    const uint channel_idx = grid_idx.y;

    if (channel_idx >= model_dim || token_idx >= suffix_len) {
        return;
    }

    const int tap_count = max(kernel_size - 1, 0);
    const size_t state_offset = size_t(channel_idx) * state_stride;
    const device T* w_row = w + size_t(channel_idx) * size_t(kernel_size);
    const bool has_bias = b != nullptr;

    size_t in_proj_idx = size_t(token_idx) * in_proj_stride + size_t(channel_idx);
    float pre_conv_gate = float(in_proj[in_proj_idx]);
    float post_conv_gate = float(in_proj[in_proj_idx + model_dim]);
    float x_val = float(in_proj[in_proj_idx + 2 * model_dim]);

    float gated_input = x_val * pre_conv_gate;

    float acc = has_bias ? float(b[channel_idx]) : 0.0f;

    for (int tap = 0; tap < tap_count; ++tap) {
        float sample = float(state[state_offset + size_t(tap)]);
        acc += float(w_row[tap]) * sample;
    }

    acc += float(w_row[tap_count]) * gated_input;

    float gated_output = acc * post_conv_gate;

    size_t out_idx = size_t(token_idx) * model_dim + size_t(channel_idx);
    out[out_idx] = static_cast<T>(gated_output);

    if (tap_count > 0) {
        for (int tap = 0; tap < tap_count - 1; ++tap) {
            next_state[state_offset + size_t(tap)] = state[state_offset + size_t(tap + 1)];
        }
        next_state[state_offset + size_t(tap_count - 1)] = static_cast<T>(gated_input);
    }
}

#define instantiate_short_conv_prefill_kernel(type_name, type)                   \
    template [[host_name("short_conv_prefill_kernel_" #type_name)]]              \
    kernel void short_conv_prefill_kernel<type>(                                 \
        device const type* x [[buffer(0)]],                                      \
        device const type* w [[buffer(1)]],                                      \
        device const type* b [[buffer(2)]],                                      \
        device const type* state_in [[buffer(3)]],                               \
        device type* out [[buffer(4)]],                                          \
        device type* state_out [[buffer(5)]],                                    \
        constant const size_t& suffix_len [[buffer(6)]],                         \
        constant const int& kernel_size [[buffer(7)]],                           \
        constant const size_t& row_stride [[buffer(8)]],                         \
        constant const size_t& state_stride [[buffer(9)]],                       \
        constant const uint& num_channels [[buffer(10)]],                        \
        uint2 grid_idx [[thread_position_in_grid]]                               \
    );

#define instantiate_short_conv_decode_kernel(type_name, type)                    \
    template [[host_name("short_conv_decode_kernel_" #type_name)]]               \
    kernel void short_conv_decode_kernel<type>(                                  \
        device const type* x [[buffer(0)]],                                      \
        device const type* w [[buffer(1)]],                                      \
        device const type* b [[buffer(2)]],                                      \
        device const type* state [[buffer(3)]],                                  \
        device type* out [[buffer(4)]],                                          \
        device type* next_state [[buffer(5)]],                                   \
        constant const size_t& suffix_len [[buffer(6)]],                         \
        constant const int& kernel_size [[buffer(7)]],                           \
        constant const size_t& row_stride [[buffer(8)]],                         \
        constant const size_t& state_stride [[buffer(9)]],                       \
        constant const uint& num_channels [[buffer(10)]],                        \
        uint2 grid_idx [[thread_position_in_grid]]                               \
    );

instantiate_short_conv_prefill_kernel(float, float);
instantiate_short_conv_prefill_kernel(bfloat, bfloat);
instantiate_short_conv_prefill_kernel(half, half);

instantiate_short_conv_decode_kernel(float, float);
instantiate_short_conv_decode_kernel(bfloat, bfloat);
instantiate_short_conv_decode_kernel(half, half);

#undef instantiate_short_conv_prefill_kernel
#undef instantiate_short_conv_decode_kernel

