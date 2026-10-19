#include <metal_stdlib>
#include "../common/dsl.h"

using namespace metal;

// Saves, for every token tile, the KERNEL_SIZE - 1 rows that precede it, before the in-place DeltaNetConvScan
// overwrites them: packed[tile, tap, channel] = row (tile * tile_tokens - TAPS + tap) of [conv state ++ x].
// Tile 0 takes the carried conv state; later tiles take raw x rows (x is read before the scan runs).
//
// state_in: [conv_dim, KERNEL_SIZE - 1]
// x:        [suffix_len, row_stride]
// packed:   [num_tiles, KERNEL_SIZE - 1, conv_dim]
template <typename T, uint KERNEL_SIZE>
VARIANTS(T, float, bfloat)
VARIANTS(KERNEL_SIZE, 4)
PUBLIC KERNEL(DeltaNetConvPack)(
    device const float* state_in,
    device const T* x,
    device float* packed,
    constant const uint& suffix_len,
    constant const uint& tile_tokens,
    constant const uint& row_stride,
    constant const uint& conv_dim,
    const uint channel_idx AXIS(conv_dim, 128),
    const uint tile_idx AXIS(suffix_len.div_ceil(tile_tokens), 1)
) {
  constexpr uint TAPS = KERNEL_SIZE - 1;
  const size_t first_row = size_t(tile_idx) * tile_tokens;
  for (uint tap = 0; tap < TAPS; ++tap) {
    float value;
    if (tile_idx == 0) {
      value = state_in[channel_idx * TAPS + tap];
    } else {
      value = float(x[(first_row - TAPS + tap) * row_stride + channel_idx]);
    }
    packed[(tile_idx * TAPS + tap) * size_t(conv_dim) + channel_idx] = value;
  }
}
