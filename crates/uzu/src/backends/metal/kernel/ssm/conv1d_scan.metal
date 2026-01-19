#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

constant int activation_type [[function_constant(0)]];
constant bool has_bias [[function_constant(1)]];

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
  return static_cast<T>(
      0.5f * xf *
      (1.0f + fast::tanh(0.797885f * (xf + 0.044715f * xf * xf * xf)))
  );
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
[[kernel, max_total_threads_per_threadgroup(32)]]
void conv1d_pack_prefix_kernel(
    device const T* state_in [[buffer(0)]],
    device const T* x [[buffer(1)]],
    device T* padded [[buffer(2)]],
    constant const size_t& state_stride [[buffer(3)]],
    constant const size_t& row_stride [[buffer(4)]],
    constant const size_t& suffix_len [[buffer(5)]],
    constant const uint& num_channels [[buffer(6)]],
    uint2 grid_idx [[thread_position_in_grid]]
) {
  const uint channel_idx = grid_idx.x;
  const uint row_idx = grid_idx.y;
  const size_t padded_rows = state_stride + suffix_len;
  if (channel_idx >= num_channels || row_idx >= padded_rows) {
    return;
  }
  const size_t padded_index =
      size_t(row_idx) * row_stride + size_t(channel_idx);
  if (row_idx < state_stride) {
    const size_t state_index =
        size_t(channel_idx) * state_stride + size_t(row_idx);
    padded[padded_index] = state_in[state_index];
  } else {
    const size_t token = row_idx - state_stride;
    const size_t x_index = size_t(token) * row_stride + size_t(channel_idx);
    padded[padded_index] = x[x_index];
  }
}

template <typename T>
[[kernel, max_total_threads_per_threadgroup(32)]]
void conv1d_scan_kernel(
    device const T* padded [[buffer(0)]], // (prefix+suffix, channels)
    device const T* w [[buffer(1)]],      // (channels, kernel)
    device const T* b [[buffer(2), function_constant(has_bias)]], // optional (channels)
    device T* x_out [[buffer(3)]],        // (suffix, inner_dim)
    device T* b_out [[buffer(4)]],        // (suffix, proj_dim)
    device T* c_out [[buffer(5)]],        // (suffix, proj_dim)
    device T* state_out [[buffer(6)]],    // (channels, kernel-1)
    constant const size_t& suffix_len [[buffer(7)]],
    constant const int& kernel_size [[buffer(8)]],
    constant const size_t& row_stride [[buffer(9)]],
    constant const size_t& state_stride [[buffer(10)]],
    constant const uint& num_channels [[buffer(11)]],
    constant const uint& inner_dim [[buffer(12)]],
    constant const uint& proj_dim [[buffer(13)]],
    uint3 grid_idx [[thread_position_in_grid]]
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

  if (token_idx < suffix_len) {
    float acc = 0.0f;
    if (has_bias) {
      acc = float(b[channel_idx]);
    }
    for (int tap = 0; tap < kernel_value; ++tap) {
      const size_t padded_idx = size_t(token_idx) + size_t(tap);
      float sample = 0.0f;
      const size_t padded_index = size_t(padded_idx) * row_stride + channel_idx;
      sample = float(padded[padded_index]);
      acc += float(w_row[tap]) * sample;
    }

    const T activated =
        apply_activation_fn(static_cast<T>(acc), activation_type);
    if (channel_idx < inner_dim) {
      const size_t dst = size_t(token_idx) * inner_dim + channel_idx;
      x_out[dst] = activated;
    } else if (channel_idx < inner_dim + proj_dim) {
      const size_t dst =
          size_t(token_idx) * proj_dim + (channel_idx - inner_dim);
      b_out[dst] = activated;
    } else if (channel_idx < inner_dim + 2 * proj_dim) {
      const size_t dst =
          size_t(token_idx) * proj_dim + (channel_idx - inner_dim - proj_dim);
      c_out[dst] = activated;
    }
  } else if (tap_count > 0) {
    const size_t tap = size_t(token_idx - suffix_len);
    if (tap >= size_t(tap_count)) {
      return;
    }
    const size_t padded_index = size_t(token_idx) * row_stride + channel_idx;
    const float sample = float(padded[padded_index]);
    state_out[state_offset + tap] = static_cast<T>(sample);
  }
}

template <typename T>
[[kernel, max_total_threads_per_threadgroup(32)]]
void conv1d_decode_kernel(
    device const T* x [[buffer(0)]],
    device const T* w [[buffer(1)]],
    device const T* b [[buffer(2), function_constant(has_bias)]],
    device const T* state [[buffer(3)]],
    device T* x_out [[buffer(4)]],
    device T* b_out [[buffer(5)]],
    device T* c_out [[buffer(6)]],
    device T* next_state [[buffer(7)]],
    constant const int& kernel_size [[buffer(8)]],
    constant const size_t& row_stride [[buffer(9)]],
    constant const size_t& state_stride [[buffer(10)]],
    constant const uint& num_channels [[buffer(11)]],
    constant const size_t& suffix_len [[buffer(12)]],
    constant const uint& inner_dim [[buffer(13)]],
    constant const uint& proj_dim [[buffer(14)]],
    uint2 grid_idx [[thread_position_in_grid]]
) {
  const int kernel_value = kernel_size;
  if (kernel_value <= 0) {
    return;
  }

  const uint token_idx = grid_idx.x;
  const uint channel_idx = grid_idx.y;
  if (channel_idx >= num_channels || token_idx >= suffix_len) {
    return;
  }

  const size_t x_idx = size_t(token_idx) * row_stride + size_t(channel_idx);
  const size_t state_offset = size_t(channel_idx) * state_stride;
  const device T* w_row = w + size_t(channel_idx) * size_t(kernel_value);

  float acc = 0.0f;
  if (has_bias) {
    acc = float(b[channel_idx]);
  }
  const int state_taps = max(kernel_value - 1, 0);
  for (int tap = 0; tap < state_taps; ++tap) {
    const size_t state_index = state_offset + size_t(tap);
    const float sample = float(state[state_index]);
    acc += float(w_row[tap]) * sample;
  }

  const float current = float(x[x_idx]);
  const int current_tap_index = state_taps;
  if (current_tap_index < kernel_value) {
    acc += float(w_row[current_tap_index]) * current;
  }

  const T activated = apply_activation_fn(static_cast<T>(acc), activation_type);
  if (channel_idx < inner_dim) {
    const size_t dst = size_t(token_idx) * inner_dim + channel_idx;
    x_out[dst] = activated;
  } else if (channel_idx < inner_dim + proj_dim) {
    const size_t dst = size_t(token_idx) * proj_dim + (channel_idx - inner_dim);
    b_out[dst] = activated;
  } else if (channel_idx < inner_dim + 2 * proj_dim) {
    const size_t dst =
        size_t(token_idx) * proj_dim + (channel_idx - inner_dim - proj_dim);
    c_out[dst] = activated;
  }

  if (state_taps == 0) {
    return;
  }

  for (int tap = 0; tap < state_taps - 1; ++tap) {
    const size_t dst_index = state_offset + size_t(tap);
    const size_t src_index = state_offset + size_t(tap + 1);
    next_state[dst_index] = state[src_index];
  }

  const size_t tail_index = state_offset + size_t(state_taps - 1);
  next_state[tail_index] = static_cast<T>(current);
}

#define instantiate_conv1d_pack_prefix_kernel(type_name, type)                 \
  template [[host_name("conv1d_pack_prefix_kernel_" #type_name)]]              \
  kernel void conv1d_pack_prefix_kernel<type>(                                 \
      device const type* state_in [[buffer(0)]],                               \
      device const type* x [[buffer(1)]],                                      \
      device type* padded [[buffer(2)]],                                       \
      constant const size_t& state_stride [[buffer(3)]],                       \
      constant const size_t& row_stride [[buffer(4)]],                         \
      constant const size_t& suffix_len [[buffer(5)]],                         \
      constant const uint& num_channels [[buffer(6)]],                         \
      uint2 grid_idx [[thread_position_in_grid]]                               \
  );

#define instantiate_conv1d_scan_kernel(type_name, type)                        \
  template [[host_name("conv1d_scan_kernel_" #type_name)]]                     \
  kernel void conv1d_scan_kernel<type>(                                        \
      device const type* padded [[buffer(0)]],                                 \
      device const type* w [[buffer(1)]],                                      \
      device const type* b [[buffer(2), function_constant(has_bias)]],         \
      device type* x_out [[buffer(3)]],                                        \
      device type* b_out [[buffer(4)]],                                        \
      device type* c_out [[buffer(5)]],                                        \
      device type* state_out [[buffer(6)]],                                    \
      constant const size_t& suffix_len [[buffer(7)]],                         \
      constant const int& kernel_size [[buffer(8)]],                           \
      constant const size_t& row_stride [[buffer(9)]],                         \
      constant const size_t& state_stride [[buffer(10)]],                      \
      constant const uint& num_channels [[buffer(11)]],                        \
      constant const uint& inner_dim [[buffer(12)]],                           \
      constant const uint& proj_dim [[buffer(13)]],                            \
      uint3 grid_idx [[thread_position_in_grid]]                               \
  );

instantiate_conv1d_pack_prefix_kernel(float, float);
instantiate_conv1d_pack_prefix_kernel(bfloat, bfloat);
instantiate_conv1d_pack_prefix_kernel(half, half);

instantiate_conv1d_scan_kernel(float, float);
instantiate_conv1d_scan_kernel(bfloat, bfloat);
instantiate_conv1d_scan_kernel(half, half);

#define instantiate_conv1d_decode_kernel(type_name, type)                      \
  template [[host_name("conv1d_decode_kernel_" #type_name)]]                   \
  kernel void conv1d_decode_kernel<type>(                                      \
      device const type* x [[buffer(0)]],                                      \
      device const type* w [[buffer(1)]],                                      \
      device const type* b [[buffer(2), function_constant(has_bias)]],         \
      device const type* state [[buffer(3)]],                                  \
      device type* x_out [[buffer(4)]],                                        \
      device type* b_out [[buffer(5)]],                                        \
      device type* c_out [[buffer(6)]],                                        \
      device type* next_state [[buffer(7)]],                                   \
      constant const int& kernel_size [[buffer(8)]],                           \
      constant const size_t& row_stride [[buffer(9)]],                         \
      constant const size_t& state_stride [[buffer(10)]],                      \
      constant const uint& num_channels [[buffer(11)]],                        \
      constant const size_t& suffix_len [[buffer(12)]],                        \
      constant const uint& inner_dim [[buffer(13)]],                           \
      constant const uint& proj_dim [[buffer(14)]],                            \
      uint2 grid_idx [[thread_position_in_grid]]                               \
  );

#undef instantiate_conv1d_pack_prefix_kernel
#undef instantiate_conv1d_scan_kernel
instantiate_conv1d_decode_kernel(float, float);
instantiate_conv1d_decode_kernel(bfloat, bfloat);
instantiate_conv1d_decode_kernel(half, half);
#undef instantiate_conv1d_decode_kernel
