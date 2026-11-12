import definitions;

[[Variants("T", "float, bf16")]]
[[Kernel]] [[Public]]
[[Precondition("new", "KERNEL_SIZE == 4")]]
[[Precondition("encode", "suffix_len <= tile_tokens || tile_tokens >= KERNEL_SIZE - 1")]]
func DeltaNetConvPack<T: IStorageFloat>(
  Ptr<float, Access.Read> state_in,
  Ptr<T, Access.Read> x,
  Ptr<float, Access.ReadWrite> packed,
  uint suffix_len,
  uint tile_tokens,
  uint row_stride,
  uint conv_dim,
  [[Axis("conv_dim", "128")]] uint channel,
  [[Axis("suffix_len.div_ceil(tile_tokens)", "1")]] uint tile,
  [[Specialize]] uint KERNEL_SIZE,
) {
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
