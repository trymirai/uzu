import definitions;
import storage;

[[Variants("T", "float, half, BFloat16")]]
[[Public]] [[Kernel]] func DeltaNetConvScan<T : ITensorElement>(
    StructuredBuffer<float> conv_padded,
    StructuredBuffer<float> conv_weight,
    [[Optional("has_bias")]] StructuredBuffer<float> bias,
    RWStructuredBuffer<Atomic<uint>> in_proj,
    RWStructuredBuffer<float> state_out,
    uint suffix_len,
    uint kernel_size,
    uint row_stride,
    uint state_stride,
    uint conv_dim,
    uint out_stride,
    [[Specialize]] bool has_bias,
    [[Groups("suffix_len.div_ceil(4).max(1)")]] uint token_idx,
    [[Groups("conv_dim.div_ceil(128)")]] uint channel_group,
    [[Threads("128")]] uint channel_lane,
) {
  uint3 gid = uint3(channel_group * 128 + channel_lane, token_idx, 0);

  let channel = gid.x;
  if (channel >= conv_dim) {
    return;
  }
  let tile = gid.y + gid.z * 65535u;
  // Tile zero also preserves the state-copy behavior for an empty suffix.
  if (tile != 0u && (suffix_len == 0u || tile > (suffix_len - 1u) / 4u))
    return;
  let first = tile * 4u;
  [ForceUnroll] for (uint local = 0; local < 4u; local++) {
    if (local >= suffix_len - first)
      break;
    let token = first + local;
    var sum = 0.0;
    if (has_bias) {
      sum = bias[channel];
    }
    for (var tap = 0u; tap < kernel_size; tap++) {
      sum += conv_weight[channel * kernel_size + tap] * conv_padded[(token + tap) * row_stride + channel];
    }
    tensorStore<T>(in_proj, token * out_stride + channel, activate(sum, 0u, 1.0));
  }
  if (tile == 0u)
    for (var tap = 0u; tap < state_stride; tap++) {
      state_out[channel * state_stride + tap] = conv_padded[(suffix_len + tap) * row_stride + channel];
    }
}
