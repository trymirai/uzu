import definitions;
import storage;

[[Variants("T", "float, half, BFloat16")]]
[[Public]] [[Kernel]] func DeltaNetConvUpdate<T : ITensorElement>(
    StructuredBuffer<float> conv_weight,
    [[Optional("has_bias")]] StructuredBuffer<float> bias,
    RWStructuredBuffer<Atomic<uint>> in_out,
    RWStructuredBuffer<float> state,
    uint kernel_size,
    uint conv_dim,
    uint state_stride,
    [[Specialize]] bool has_bias,
    [[Axis("conv_dim", "256")]] uint channel_idx,
) {
  uint3 gid = uint3(channel_idx, 0, 0);

  let channel = gid.x;
  if (channel >= conv_dim) {
    return;
  }
  let base = channel * state_stride;
  let weight = channel * kernel_size;
  let x = tensorLoad<T>(in_out, channel);
  var sum = 0.0;
  if (has_bias) {
    sum = bias[channel];
  }
  for (var tap = 0u; tap + 1u < kernel_size; tap++) {
    sum += state[base + tap] * conv_weight[weight + tap];
  }
  sum += x * conv_weight[weight + kernel_size - 1u];
  tensorStore<T>(in_out, channel, activate(sum, 0u, 1.0));
  for (var tap = 1u; tap + 1u < kernel_size; tap++) {
    state[base + tap - 1u] = state[base + tap];
  }
  if (kernel_size > 1u) {
    state[base + kernel_size - 2u] = x;
  }
}
