#include <metal_stdlib>
#include "../definitions.metal"

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(ShortConvPack)(
    device const T* state_in,
    device const T* in_proj,
    device T* padded,
    constant const uint& state_stride,
    constant const uint& suffix_len,
    constant const uint& in_proj_stride,
    constant const uint& model_dim,
    const uint channel_idx AXIS(suffix_len, 32),
    const uint row_idx AXIS(model_dim, 1)
) {
  const uint padded_rows = state_stride + suffix_len;
  if (channel_idx >= model_dim || row_idx >= padded_rows) {
    return;
  }

  const uint padded_offset = row_idx * model_dim + channel_idx;
  if (row_idx < state_stride) {
    // Copy from state
    const uint state_idx = channel_idx * state_stride + row_idx;
    padded[padded_offset] = state_in[state_idx];
  } else {
    // Compute gated input from in_proj
    const uint token = row_idx - state_stride;
    const uint in_proj_idx = token * in_proj_stride + channel_idx;

    float pre_gate = float(in_proj[in_proj_idx]);
    float x_in = float(in_proj[in_proj_idx + 2 * model_dim]);
    float x = x_in * pre_gate;

    padded[padded_offset] = static_cast<T>(x);
  }
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(ShortConvPrefill)(
    device const T* padded,
    device const T* in_proj,
    device const T* w,
    device const T* b OPTIONAL(has_bias),
    device T* out,
    device T* state_out,
    constant const uint& suffix_len,
    constant const uint& kernel_size,
    constant const uint& in_proj_stride,
    constant const uint& state_stride,
    constant const uint& model_dim,
    const bool has_bias SPECIALIZE,
    const uint token_idx AXIS(suffix_len + kernel_size.saturating_sub(1), 32),
    const uint channel_idx AXIS(model_dim, 1)
) {
  const uint tap_count = kernel_size > 0 ? kernel_size - 1 : 0u;
  const device T* w_row = w + channel_idx * kernel_size;

  // Threads [0..suffix_len-1]: Compute outputs
  if (token_idx < suffix_len) {
    float acc = 0.0f;
    if (has_bias) {
      acc = float(b[channel_idx]);
    }

    // Convolve using padded buffer
    for (uint tap = 0; tap < kernel_size; ++tap) {
      const uint padded_row = token_idx + tap;
      const uint padded_offset = padded_row * model_dim + channel_idx;
      float sample = float(padded[padded_offset]);
      acc += float(w_row[tap]) * sample;
    }

    // Apply post-gate from in_proj
    const uint in_proj_idx = token_idx * in_proj_stride + channel_idx;
    float post_conv_gate = float(in_proj[in_proj_idx + model_dim]);
    float gated_output = acc * post_conv_gate;

    // Write output
    const uint out_idx = token_idx * model_dim + channel_idx;
    out[out_idx] = static_cast<T>(gated_output);
  }
  // Threads [suffix_len..work_len-1]: Write state
  else if (tap_count > 0) {
    const uint tap = token_idx - suffix_len;
    if (tap >= tap_count) {
      return;
    }

    // Copy last tap_count values from padded to state_out
    const uint padded_row = suffix_len + tap;
    const uint padded_offset = padded_row * model_dim + channel_idx;
    const uint state_idx = channel_idx * state_stride + tap;

    state_out[state_idx] = padded[padded_offset];
  }
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(ShortConvDecode)(
    device const T* in_proj,
    device const T* w,
    device const T* b OPTIONAL(has_bias),
    device const T* state,
    device T* out,
    device T* next_state,
    constant const uint& suffix_len,
    constant const uint& kernel_size,
    constant const uint& in_proj_stride,
    constant const uint& state_stride,
    constant const uint& model_dim,
    const bool has_bias SPECIALIZE,
    const uint token_idx AXIS(suffix_len, 32),
    const uint channel_idx AXIS(model_dim, 1)
) {
  const uint tap_count = kernel_size > 0 ? kernel_size - 1 : 0u;
  const uint state_offset = channel_idx * state_stride;
  const device T* w_row = w + channel_idx * kernel_size;

  uint in_proj_idx = token_idx * in_proj_stride + channel_idx;
  float pre_conv_gate = float(in_proj[in_proj_idx]);
  float post_conv_gate = float(in_proj[in_proj_idx + model_dim]);
  float x_in = float(in_proj[in_proj_idx + 2 * model_dim]);

  float x = x_in * pre_conv_gate;

  float acc = 0.0f;
  if (has_bias) {
    acc = float(b[channel_idx]);
  }

  for (uint tap = 0; tap < tap_count; ++tap) {
    float sample = float(state[state_offset + tap]);
    acc += float(w_row[tap]) * sample;
  }

  acc += float(w_row[tap_count]) * x;

  float gated_output = acc * post_conv_gate;

  uint out_idx = token_idx * model_dim + channel_idx;
  out[out_idx] = static_cast<T>(gated_output);

  if (tap_count > 0) {
    for (uint tap = 0; tap < tap_count - 1; ++tap) {
      next_state[state_offset + tap] = state[state_offset + tap + 1];
    }
    next_state[state_offset + tap_count - 1] = static_cast<T>(x);
  }
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(ShortConvTrie)(
    device const T* in_proj,
    device const T* w,
    device const T* b OPTIONAL(has_bias),
    device const T* base_state,
    device const int* parents,
    device T* out,
    device T* suffix_state,
    constant const uint& suffix_len,
    constant const uint& kernel_size,
    constant const uint& in_proj_stride,
    constant const uint& state_stride,
    constant const uint& model_dim,
    const bool has_bias SPECIALIZE,
    const uint channel_idx AXIS(model_dim, 32)
) {
  const uint tap_count = kernel_size > 0 ? kernel_size - 1 : 0u;
  const device T* w_row = w + channel_idx * kernel_size;
  const uint base_state_offset = channel_idx * state_stride;

  for (uint node = 0; node < suffix_len; ++node) {
    // Read gated input from in_proj
    const uint in_proj_idx = node * in_proj_stride + channel_idx;
    float pre_conv_gate = float(in_proj[in_proj_idx]);
    float post_conv_gate = float(in_proj[in_proj_idx + model_dim]);
    float x_in = float(in_proj[in_proj_idx + 2 * model_dim]);
    float x = x_in * pre_conv_gate;

    // Select parent state (root uses base_state)
    const int parent = parents[node];
    const device T* parent_state =
        (parent < 0)
            ? (base_state + base_state_offset)
            : (suffix_state + (node * model_dim + channel_idx) * state_stride);

    float acc = 0.0f;
    if (has_bias) {
      acc = float(b[channel_idx]);
    }

    for (uint tap = 0; tap < tap_count; ++tap) {
      float sample = float(parent_state[size_t(tap)]);
      acc += float(w_row[tap]) * sample;
    }

    acc += float(w_row[tap_count]) * x;

    // Apply post-gate
    float gated_output = acc * post_conv_gate;

    // Write output
    out[node * model_dim + channel_idx] = static_cast<T>(gated_output);

    // Write post-state for this node
    if (tap_count > 0) {
      device T* dst_state =
          suffix_state + (node * model_dim + channel_idx) * state_stride;
      for (uint tap = 0; tap < tap_count - 1; ++tap) {
        dst_state[tap] = parent_state[tap + 1];
      }
      dst_state[tap_count - 1] = static_cast<T>(x);
    }
  }
}