#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

template <typename T>
kernel void short_conv_pack_kernel(
    device const T* state_in [[buffer(0)]],
    device const T* in_proj [[buffer(1)]],
    device T* padded [[buffer(2)]],
    constant const size_t& state_stride [[buffer(3)]],
    constant const size_t& suffix_len [[buffer(4)]],
    constant const size_t& in_proj_stride [[buffer(5)]],
    constant const uint& model_dim [[buffer(6)]],
    uint2 grid_idx [[thread_position_in_grid]]
) {
    const uint channel_idx = grid_idx.x;
    const uint row_idx = grid_idx.y;
    const size_t padded_rows = state_stride + suffix_len;

    if (channel_idx >= model_dim || row_idx >= padded_rows) {
        return;
    }

    const size_t padded_idx = size_t(row_idx) * model_dim + size_t(channel_idx);

    if (row_idx < state_stride) {
        // Copy from state
        const size_t state_idx = size_t(channel_idx) * state_stride + size_t(row_idx);
        padded[padded_idx] = state_in[state_idx];
    } else {
        // Compute gated input from in_proj
        const size_t token = row_idx - state_stride;
        const size_t in_proj_idx = size_t(token) * in_proj_stride + size_t(channel_idx);

        float pre_gate = float(in_proj[in_proj_idx]);
        float x = float(in_proj[in_proj_idx + 2 * model_dim]);
        float gated_input = x * pre_gate;

        padded[padded_idx] = static_cast<T>(gated_input);
    }
}

template <typename T>
kernel void short_conv_prefill_kernel(
    device const T* padded [[buffer(0)]],
    device const T* in_proj [[buffer(1)]],
    device const T* w [[buffer(2)]],
    device const T* b [[buffer(3)]],
    device T* out [[buffer(4)]],
    device T* state_out [[buffer(5)]],
    constant const size_t& suffix_len [[buffer(6)]],
    constant const int& kernel_size [[buffer(7)]],
    constant const size_t& in_proj_stride [[buffer(8)]],
    constant const size_t& state_stride [[buffer(9)]],
    constant const uint& model_dim [[buffer(10)]],
    uint2 grid_idx [[thread_position_in_grid]]
) {
    const int tap_count = max(kernel_size - 1, 0);
    const size_t work_len = suffix_len + size_t(tap_count);

    const uint token_idx = grid_idx.x;
    const uint channel_idx = grid_idx.y;

    if (channel_idx >= model_dim || token_idx >= work_len) {
        return;
    }

    const device T* w_row = w + size_t(channel_idx) * size_t(kernel_size);
    const bool has_bias = b != nullptr;

    // Threads [0..suffix_len-1]: Compute outputs
    if (token_idx < suffix_len) {
        float acc = has_bias ? float(b[channel_idx]) : 0.0f;

        // Convolve using padded buffer
        for (int tap = 0; tap < kernel_size; ++tap) {
            const size_t padded_idx = size_t(token_idx + tap);
            const size_t padded_index = padded_idx * model_dim + channel_idx;
            float sample = float(padded[padded_index]);
            acc += float(w_row[tap]) * sample;
        }

        // Apply post-gate from in_proj
        const size_t in_proj_idx = size_t(token_idx) * in_proj_stride + size_t(channel_idx);
        float post_conv_gate = float(in_proj[in_proj_idx + model_dim]);
        float gated_output = acc * post_conv_gate;

        // Write output
        const size_t out_idx = size_t(token_idx) * model_dim + size_t(channel_idx);
        out[out_idx] = static_cast<T>(gated_output);
    }
    // Threads [suffix_len..work_len-1]: Write state
    else if (tap_count > 0) {
        const size_t tap = size_t(token_idx - suffix_len);
        if (tap >= size_t(tap_count)) {
            return;
        }

        // Copy last tap_count values from padded to state_out
        const size_t padded_idx = suffix_len + tap;
        const size_t padded_index = padded_idx * model_dim + channel_idx;
        const size_t state_idx = size_t(channel_idx) * state_stride + tap;

        state_out[state_idx] = padded[padded_index];
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

// ============================================================================
// Template instantiations
// ============================================================================
#define instantiate_short_conv_pack_kernel(type_name, type)                    \
    template [[host_name("short_conv_pack_kernel_" #type_name)]]               \
    kernel void short_conv_pack_kernel<type>(                                  \
        device const type* state_in [[buffer(0)]],                             \
        device const type* in_proj [[buffer(1)]],                              \
        device type* padded [[buffer(2)]],                                     \
        constant const size_t& state_stride [[buffer(3)]],                     \
        constant const size_t& suffix_len [[buffer(4)]],                       \
        constant const size_t& in_proj_stride [[buffer(5)]],                   \
        constant const uint& model_dim [[buffer(6)]],                          \
        uint2 grid_idx [[thread_position_in_grid]]                             \
    );

#define instantiate_short_conv_prefill_kernel(type_name, type)                 \
    template [[host_name("short_conv_prefill_kernel_" #type_name)]]            \
    kernel void short_conv_prefill_kernel<type>(                               \
        device const type* padded [[buffer(0)]],                               \
        device const type* in_proj [[buffer(1)]],                              \
        device const type* w [[buffer(2)]],                                    \
        device const type* b [[buffer(3)]],                                    \
        device type* out [[buffer(4)]],                                        \
        device type* state_out [[buffer(5)]],                                  \
        constant const size_t& suffix_len [[buffer(6)]],                       \
        constant const int& kernel_size [[buffer(7)]],                         \
        constant const size_t& in_proj_stride [[buffer(8)]],                   \
        constant const size_t& state_stride [[buffer(9)]],                     \
        constant const uint& model_dim [[buffer(10)]],                         \
        uint2 grid_idx [[thread_position_in_grid]]                             \
    );

#define instantiate_short_conv_decode_kernel(type_name, type)                  \
    template [[host_name("short_conv_decode_kernel_" #type_name)]]             \
    kernel void short_conv_decode_kernel<type>(                                \
        device const type* x [[buffer(0)]],                                    \
        device const type* w [[buffer(1)]],                                    \
        device const type* b [[buffer(2)]],                                    \
        device const type* state [[buffer(3)]],                                \
        device type* out [[buffer(4)]],                                        \
        device type* next_state [[buffer(5)]],                                 \
        constant const size_t& suffix_len [[buffer(6)]],                       \
        constant const int& kernel_size [[buffer(7)]],                         \
        constant const size_t& row_stride [[buffer(8)]],                       \
        constant const size_t& state_stride [[buffer(9)]],                     \
        constant const uint& num_channels [[buffer(10)]],                      \
        uint2 grid_idx [[thread_position_in_grid]]                             \
    );

instantiate_short_conv_pack_kernel(float, float);
instantiate_short_conv_pack_kernel(bfloat, bfloat);
instantiate_short_conv_pack_kernel(half, half);

instantiate_short_conv_prefill_kernel(float, float);
instantiate_short_conv_prefill_kernel(bfloat, bfloat);
instantiate_short_conv_prefill_kernel(half, half);

instantiate_short_conv_decode_kernel(float, float);
instantiate_short_conv_decode_kernel(bfloat, bfloat);
instantiate_short_conv_decode_kernel(half, half);

#undef instantiate_short_conv_pack_kernel
#undef instantiate_short_conv_prefill_kernel
#undef instantiate_short_conv_decode_kernel
