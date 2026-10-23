import definitions;

// The KERNEL_SIZE - 1 rows before each token tile, which the in-place DeltaNetConvScan overwrites: packed[tile, tap, c]
// is row tile tile_tokens - (KERNEL_SIZE - 1) + tap of [conv state ++ x], so tile 0 copies the state row of channel c
// and later tiles convert x through FP32. The precondition reproduces the CPU's failure when a tile spans fewer rows.

static const uint DELTA_NET_CONV_PACK_THREADS = 128;

[[Kernel]] [[Public]]
[[Precondition("new", "KERNEL_SIZE == 4")]]
[[Precondition("encode", "suffix_len <= tile_tokens || tile_tokens >= KERNEL_SIZE - 1")]]
func DeltaNetConvPack<T: IModelFloat>(
  Ptr<float, Access.Read> state_in,
  Ptr<T, Access.Read> x,
  Ptr<float, Access.ReadWrite> packed,
  uint suffix_len,
  uint tile_tokens,
  uint row_stride,
  uint conv_dim,
  [[Groups("conv_dim.div_ceil(128)")]] uint channel_group,
  [[Groups("suffix_len.div_ceil(tile_tokens)")]] uint tile,
  [[Threads("128")]] uint thread_idx,
  [[Specialize]] uint KERNEL_SIZE,
) {
  let channel = channel_group * DELTA_NET_CONV_PACK_THREADS + thread_idx;
  if (channel < conv_dim) {
    let taps = KERNEL_SIZE - 1;
    for (uint tap = 0; tap < taps; tap++) {
      var value = 0.0;
      if (tile == 0) {
        value = state_in[channel * taps + tap];
      } else {
        value = x[(tile * tile_tokens - taps + tap) * row_stride + channel].toFloat();
      }
      packed[(tile * taps + tap) * conv_dim + channel] = value;
    }
  }
}
