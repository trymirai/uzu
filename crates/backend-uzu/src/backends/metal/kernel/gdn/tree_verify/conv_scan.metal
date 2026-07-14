#include <metal_stdlib>
#include "../../activation/activations.h"
#include "../../common/dsl.h"

using namespace metal;

// Applies the short convolution independently along every proposal-tree path.
//
// Shapes (T = suffix_len, K = kernel_size):
//   in_proj     [T, total_proj_dim] T
//   conv_weight [conv_dim, K] f32
//   bias        [conv_dim] f32, optional
//   base_state  [conv_dim, K - 1] f32, read-only
//   parents     [T] i32
//   out_proj    [T, total_proj_dim] T
//   suffix_state [T, conv_dim, K - 1] f32
//
// One thread owns one (tree node, projection channel). Convolved channels walk
// the node's parent chain and then committed history, recording the resulting
// per-node window for later acceptance. Other projection channels pass through.
template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(DeltaNetConvTreeScan)(
    device const T* in_proj,
    device const float* conv_weight,
    device const float* bias OPTIONAL(has_bias),
    device const float* base_state,
    device const int* parents,
    device T* out_proj,
    device float* suffix_state,
    constant const uint& suffix_len,
    const uint kernel_size SPECIALIZE,
    constant const uint& total_proj_dim,
    constant const uint& conv_dim,
    const bool has_bias SPECIALIZE,
    const uint node_idx AXIS(suffix_len, 1),
    const uint channel_idx AXIS(total_proj_dim, 256)
) {
  const uint proj_idx = node_idx * total_proj_dim + channel_idx;
  if (channel_idx >= conv_dim) {
    out_proj[proj_idx] = in_proj[proj_idx];
    return;
  }

  const uint state_stride = kernel_size - 1;
  float acc = has_bias ? bias[channel_idx] : 0.0f;
  const uint weight_offset = channel_idx * kernel_size;
  const uint base_state_offset = channel_idx * state_stride;
  int source_row = int(node_idx);
  for (uint history_offset = 0; history_offset < kernel_size; ++history_offset) {
    float sample;
    if (source_row >= 0) {
      const uint source_proj_idx = uint(source_row) * total_proj_dim + channel_idx;
      sample = float(in_proj[source_proj_idx]);
    } else {
      const uint base_state_tap = state_stride - uint(-source_row);
      sample = base_state[base_state_offset + base_state_tap];
    }

    const uint weight_tap = kernel_size - 1 - history_offset;
    acc += sample * conv_weight[weight_offset + weight_tap];

    if (history_offset < state_stride) {
      const uint state_tap = state_stride - 1 - history_offset;
      const uint state_idx = (node_idx * conv_dim + channel_idx) * state_stride + state_tap;
      suffix_state[state_idx] = sample;
    }

    source_row = source_row >= 0 ? parents[source_row] : source_row - 1;
  }

  out_proj[proj_idx] = static_cast<T>(activate_silu(acc));
}
