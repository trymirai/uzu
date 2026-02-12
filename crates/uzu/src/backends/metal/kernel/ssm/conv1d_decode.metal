#include <metal_stdlib>
#include "../definitions.metal"
#include "ssm_common.h"

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