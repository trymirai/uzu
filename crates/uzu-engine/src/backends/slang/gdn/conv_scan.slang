import definitions;
import storage;

[[Variants("T", "float, half, BFloat16")]]
[[Variants("KERNEL_SIZE", "4")]]
[[Public]] [[Kernel]] func DeltaNetConvScan<T : ITensorElement, let KERNEL_SIZE : uint>(
    StructuredBuffer<float> packed,
    StructuredBuffer<float> conv_weight,
    [[Optional("has_bias")]] StructuredBuffer<float> bias,
    RWStructuredBuffer<Atomic<uint>> in_proj,
    RWStructuredBuffer<float> state_out,
    uint suffix_len,
    uint tile_tokens,
    uint conv_dim,
    uint out_stride,
    [[Specialize]] bool has_bias,
    [[Groups("suffix_len.div_ceil(tile_tokens)")]] uint tile_idx,
    [[Groups("conv_dim.div_ceil(128)")]] uint channel_group,
    [[Threads("128")]] uint channel_lane,
) {
  let channel = channel_group * 128 + channel_lane;
  if (channel >= conv_dim) {
    return;
  }

  // Each tile owns its input rows; only its initial window comes from packed.
  float window[KERNEL_SIZE - 1];
  [ForceUnroll] for (uint tap = 0; tap < KERNEL_SIZE - 1; tap++) {
    window[tap] = packed[(tile_idx * (KERNEL_SIZE - 1) + tap) * conv_dim + channel];
  }
  let first = tile_idx * tile_tokens;
  let end = min(first + tile_tokens, suffix_len);
  for (uint token = first; token < end; token++) {
    let sample = tensorLoad<T>(in_proj, token * out_stride + channel);
    var sum = has_bias ? bias[channel] : 0.0;
    [ForceUnroll] for (uint tap = 0; tap < KERNEL_SIZE - 1; tap++) {
      sum += conv_weight[channel * KERNEL_SIZE + tap] * window[tap];
    }
    sum += conv_weight[channel * KERNEL_SIZE + KERNEL_SIZE - 1] * sample;
    tensorStore<T>(in_proj, token * out_stride + channel, activate(sum, 0u, 1.0));
    [ForceUnroll] for (uint tap = 0; tap + 1 < KERNEL_SIZE - 1; tap++) {
      window[tap] = window[tap + 1];
    }
    window[KERNEL_SIZE - 2] = sample;
  }
  if (end == suffix_len) {
    [ForceUnroll] for (uint tap = 0; tap < KERNEL_SIZE - 1; tap++) {
      state_out[channel * (KERNEL_SIZE - 1) + tap] = window[tap];
    }
  }
}
