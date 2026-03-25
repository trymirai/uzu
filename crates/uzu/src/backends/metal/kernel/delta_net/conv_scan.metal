#include <metal_stdlib>
#include "../activation/activations.h"
#include "../common/dsl.h"
#include "../ssm/ssm_common.h"

using namespace metal;

// Multi-token causal conv1d with SiLU for DeltaNet.
// Reads from padded buffer (Conv1dPack output), writes conv'd+SiLU'd values
// back to in_out with stride out_stride, updates conv state.
//
// padded:    [state_stride + suffix_len, row_stride]
// w:         [conv_dim, kernel_size]
// bias:      [conv_dim] (optional)
// in_out:    [suffix_len, out_stride] — first conv_dim channels overwritten
// state_out: [conv_dim, state_stride]

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(DeltaNetConvScan)(
    device const T* padded,
    device const T* w,
    device const T* bias OPTIONAL(has_bias),
    device T* in_out,
    device T* state_out,
    constant const uint& suffix_len,
    constant const uint& kernel_size,
    constant const uint& row_stride,
    constant const uint& state_stride,
    constant const uint& conv_dim,
    constant const uint& out_stride,
    const bool has_bias SPECIALIZE,
    const uint token_idx AXIS(suffix_len + kernel_size - 1, 1),
    const uint channel_idx AXIS(conv_dim, 32)
) {
  const uint tap_count = kernel_size - 1;

  if (token_idx < suffix_len) {
    const device T* w_row = w + channel_idx * kernel_size;

    float acc = has_bias ? float(bias[channel_idx]) : 0.0f;

    for (uint tap = 0; tap < kernel_size; ++tap) {
      const uint padded_row = token_idx + tap;
      const uint padded_index = padded_row * row_stride + channel_idx;
      acc += float(w_row[tap]) * float(padded[padded_index]);
    }

    const size_t dst = size_t(token_idx) * out_stride + channel_idx;
    in_out[dst] = static_cast<T>(activate_silu(acc));
  } else {
    const uint tap = token_idx - suffix_len;
    if (tap >= tap_count) {
      return;
    }
    const uint padded_index = size_t(token_idx) * row_stride + channel_idx;
    const uint state_offset = channel_idx * state_stride;
    state_out[state_offset + tap] = padded[padded_index];
  }
}
