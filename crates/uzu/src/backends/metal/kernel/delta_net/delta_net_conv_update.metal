#include <metal_stdlib>
#include "../definitions.metal"
#include "../ssm/ssm_common.h"

using namespace metal;

// Single-token causal conv1d with SiLU, in-place.
template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(DeltaNetConvUpdate)(
    device const T* w,
    device const T* bias OPTIONAL(has_bias),
    device T* in_out,
    device T* state,
    constant const uint& kernel_size,
    constant const uint& conv_dim,
    constant const uint& state_stride,
    const bool has_bias SPECIALIZE,
    const uint channel_idx AXIS(conv_dim, 1)
) {
  if (kernel_size == 0) {
    return;
  }

  const uint tap_count = kernel_size - 1;
  const uint state_offset = channel_idx * state_stride;
  const device T* w_row = w + channel_idx * kernel_size;

  float x = float(in_out[channel_idx]);

  float acc = has_bias ? float(bias[channel_idx]) : 0.0f;

  for (uint tap = 0; tap < tap_count; ++tap) {
    acc += float(w_row[tap]) * float(state[state_offset + tap]);
  }
  acc += float(w_row[tap_count]) * x;

  in_out[channel_idx] = static_cast<T>(apply_silu(acc));

  for (uint tap = 0; tap + 1 < tap_count; ++tap) {
    state[state_offset + tap] = state[state_offset + tap + 1];
  }
  if (tap_count > 0) {
    state[state_offset + tap_count - 1] = static_cast<T>(x);
  }
}
