#include <metal_stdlib>
#include "../definitions.metal"
#include "ssm_common.h"

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(Conv1dPack)(
    device const T* state_in,
    device const T* x,
    device T* padded,
    constant const uint& state_stride,
    constant const uint& row_stride,
    constant const uint& suffix_len,
    constant const uint& num_channels,
    const uint channel_idx AXIS(num_channels, 32),
    const uint row_idx AXIS(state_stride + suffix_len, 1)
) {
  const uint padded_index = row_idx * row_stride + channel_idx;
  if (row_idx < state_stride) {
    const uint state_index = channel_idx * state_stride + row_idx;
    padded[padded_index] = state_in[state_index];
  } else {
    const uint token = row_idx - state_stride;
    const uint x_index = token * row_stride + channel_idx;
    padded[padded_index] = x[x_index];
  }
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(Conv1dDecode)(
    device const T* x,
    device const T* w,
    device const T* b OPTIONAL(has_bias),
    device const T* state,
    device T* x_out,
    device T* b_out,
    device T* c_out,
    device T* next_state,
    constant const uint& kernel_size,
    constant const uint& row_stride,
    constant const uint& state_stride,
    constant const uint& num_channels,
    constant const uint& suffix_len,
    constant const uint& inner_dim,
    constant const uint& proj_dim,
    const uint activation_type SPECIALIZE,
    const bool has_bias SPECIALIZE,
    const uint token_idx AXIS(suffix_len, 32),
    const uint channel_idx AXIS(num_channels, 1)
) {
  const uint x_idx = token_idx * row_stride + channel_idx;
  const uint state_offset = channel_idx * state_stride;
  const device T* w_row = w + channel_idx * kernel_size;

  float acc = 0.0f;
  if (has_bias) {
    acc = float(b[channel_idx]);
  }

  const uint state_taps = max(kernel_size - 1, 0u);
  for (uint tap = 0; tap < state_taps; ++tap) {
    const uint state_index = state_offset + tap;
    const float sample = float(state[state_index]);
    acc += float(w_row[tap]) * sample;
  }

  const float current = float(x[x_idx]);
  const uint current_tap_index = state_taps;
  if (current_tap_index < kernel_size) {
    acc += float(w_row[current_tap_index]) * current;
  }

  const T activated = apply_activation_fn(static_cast<T>(acc), activation_type);
  if (channel_idx < inner_dim) {
    const uint dst = token_idx * inner_dim + channel_idx;
    x_out[dst] = activated;
  } else if (channel_idx < inner_dim + proj_dim) {
    const uint dst = token_idx * proj_dim + (channel_idx - inner_dim);
    b_out[dst] = activated;
  } else if (channel_idx < inner_dim + 2 * proj_dim) {
    const uint dst =
        token_idx * proj_dim + (channel_idx - inner_dim - proj_dim);
    c_out[dst] = activated;
  }

  if (state_taps == 0) {
    return;
  }

  for (uint tap = 0; tap < state_taps - 1; ++tap) {
    const size_t dst_index = state_offset + size_t(tap);
    const size_t src_index = state_offset + size_t(tap + 1);
    next_state[dst_index] = state[src_index];
  }

  const uint tail_index = state_offset + state_taps - 1;
  next_state[tail_index] = static_cast<T>(current);
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(Conv1dScan)(
    device const T* padded, // (prefix+suffix, channels)
    device const T* w,      // (channels, kernel)
    device const T* b OPTIONAL(has_bias), // optional (channels)
    device T* x_out,              // (suffix, inner_dim)
    device T* b_out,              // (suffix, proj_dim)
    device T* c_out,              // (suffix, proj_dim)
    device T* state_out,          // (channels, kernel-1)
    constant const uint& suffix_len,
    constant const uint& kernel_size,
    constant const uint& row_stride,
    constant const uint& state_stride,
    constant const uint& num_channels,
    constant const uint& inner_dim,
    constant const uint& proj_dim,
    const uint activation_type SPECIALIZE,
    const bool has_bias SPECIALIZE,
    const uint token_idx AXIS(suffix_len + kernel_size.saturating_sub(1), 32), 
    const uint channel_idx AXIS(num_channels, 1)
) {
  const uint tap_count = max(kernel_size - 1, 0u);
  const uint state_offset = channel_idx * state_stride;
  const uint weight_offset = channel_idx * kernel_size;
  const device T* w_row = w + weight_offset;

  if (token_idx < suffix_len) {
    float acc = 0.0f;
    if (has_bias) {
      acc = float(b[channel_idx]);
    }
    for (uint tap = 0; tap < kernel_size; ++tap) {
      const uint padded_idx = size_t(token_idx) + size_t(tap);
      float sample = 0.0f;
      const uint padded_index = size_t(padded_idx) * row_stride + channel_idx;
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