#include <metal_stdlib>
#include "../common/dsl.h"

template <typename StateT, typename InputT>
VARIANTS(StateT, float, bfloat)
VARIANTS(InputT, float, bfloat)
PUBLIC KERNEL(Conv1dPack)(
    device const StateT* state_in,
    device const InputT* x,
    device StateT* padded,
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
    padded[padded_index] = static_cast<StateT>(x[x_index]);
  }
}
