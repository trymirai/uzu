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
    const uint channel_idx AXIS(model_dim, 32),
    const uint row_idx AXIS(state_stride + suffix_len, 1)
) {
  const size_t padded_offset = size_t(row_idx) * model_dim + size_t(channel_idx);

  if (row_idx < state_stride) {
    // Copy from state
    const size_t state_idx = size_t(channel_idx) * state_stride + size_t(row_idx);
    padded[padded_offset] = state_in[state_idx];
  } else {
    // Compute gated input from in_proj
    const size_t token = row_idx - state_stride;
    const size_t in_proj_idx = size_t(token) * in_proj_stride + size_t(channel_idx);

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
    device const T* b,
    device T* out,
    device T* state_out,
    const bool has_bias SPECIALIZE,
    constant const uint& suffix_len,
    constant const uint& kernel_size,
    constant const uint& in_proj_stride,
    constant const uint& state_stride,
    constant const uint& model_dim,
    const uint token_idx AXIS(suffix_len + std::cmp::max(kernel_size - 1, 0), 32),
    const uint channel_idx AXIS(model_dim, 1)
) {
  const device T* w_row = w + size_t(channel_idx) * size_t(kernel_size);
  const int tap_count = max(static_cast<int>(kernel_size) - 1, 0);

  // Threads [0..suffix_len-1]: Compute outputs
  if (token_idx < suffix_len) {
    float acc = 0.0f;
    if (has_bias) {
      acc = float(b[channel_idx]);
    }

    // Convolve using padded buffer
    for (int tap = 0; tap < kernel_size; ++tap) {
      const size_t padded_row = size_t(token_idx + tap);
      const size_t padded_offset = padded_row * model_dim + channel_idx;
      float sample = float(padded[padded_offset]);
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
    const size_t padded_row = suffix_len + tap;
    const size_t padded_offset = padded_row * model_dim + channel_idx;
    const size_t state_idx = size_t(channel_idx) * state_stride + tap;

    state_out[state_idx] = padded[padded_offset];
  }
}

template<typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(ShortConvDecode)(
    device const T* in_proj,
    device const T* w,
    device const T* b,
    device const T* state,
    device T* out,
    device T* next_state,
    const bool has_bias SPECIALIZE,
    constant const uint& suffix_len,
    constant const uint& kernel_size,
    constant const uint& in_proj_stride,
    constant const uint& state_stride,
    constant const uint& model_dim,
    const uint token_idx AXIS(suffix_len, 32),
    const uint channel_idx AXIS(model_dim, 1)
) {
  const int tap_count = max(static_cast<int>(kernel_size) - 1, 0);
  const size_t state_offset = size_t(channel_idx) * state_stride;
  const device T* w_row = w + size_t(channel_idx) * size_t(kernel_size);

  size_t in_proj_idx = size_t(token_idx) * in_proj_stride + size_t(channel_idx);
  float pre_conv_gate = float(in_proj[in_proj_idx]);
  float post_conv_gate = float(in_proj[in_proj_idx + model_dim]);
  float x_in = float(in_proj[in_proj_idx + 2 * model_dim]);

  float x = x_in * pre_conv_gate;

  float acc = 0.0f;
  if (has_bias) {
    acc = float(b[channel_idx]);
  }

  for (int tap = 0; tap < tap_count; ++tap) {
    float sample = float(state[state_offset + size_t(tap)]);
    acc += float(w_row[tap]) * sample;
  }

  acc += float(w_row[tap_count]) * x;

  float gated_output = acc * post_conv_gate;

  size_t out_idx = size_t(token_idx) * model_dim + size_t(channel_idx);
  out[out_idx] = static_cast<T>(gated_output);

  if (tap_count > 0) {
    for (int tap = 0; tap < tap_count - 1; ++tap) {
      next_state[state_offset + size_t(tap)] =
          state[state_offset + size_t(tap + 1)];
    }
    next_state[state_offset + size_t(tap_count - 1)] = static_cast<T>(x);
  }
}