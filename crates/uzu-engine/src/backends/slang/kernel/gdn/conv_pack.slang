import definitions;
import storage;

[[Variants("T", "float, half, BFloat16")]]
[[Variants("KERNEL_SIZE", "4")]]
[[Public]] [[Kernel]] func DeltaNetConvPack<T : ITensorElement, let KERNEL_SIZE : uint32_t>(
    StructuredBuffer<float> state_in,
    StructuredBuffer<uint32_t> x,
    RWStructuredBuffer<float> packed,
    uint32_t suffix_len,
    uint32_t tile_tokens,
    uint32_t row_stride,
    uint32_t conv_dim,
    [[Axis("conv_dim", "128")]] uint32_t channel_idx,
    [[Axis("suffix_len.div_ceil(tile_tokens)", "1")]] uint32_t tile_idx,
) {
  // Save the rows each tile needs before the in-place scan overwrites them.
  let taps = KERNEL_SIZE - 1;
  let first_row = tile_idx * tile_tokens;
  [ForceUnroll] for (uint tap = 0; tap < taps; tap++) {
    let value = tile_idx == 0
        ? state_in[channel_idx * taps + tap]
        : tensorLoad<T>(x, (first_row - taps + tap) * row_stride + channel_idx);
    packed[(tile_idx * taps + tap) * conv_dim + channel_idx] = value;
  }
}
