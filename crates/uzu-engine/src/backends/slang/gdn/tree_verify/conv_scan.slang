import definitions;
import storage;

[[Variants("T", "float, half, BFloat16")]]
[[Public]] [[Kernel]] func ConvTreeScan<T : ITensorElement>(
    StructuredBuffer<uint> in_proj,
    StructuredBuffer<float> conv_weight,
    [[Optional("has_bias")]] StructuredBuffer<float> bias,
    StructuredBuffer<float> base_state,
    StructuredBuffer<int> parents,
    RWStructuredBuffer<Atomic<uint>> out_proj,
    RWStructuredBuffer<float> suffix_state,
    uint suffix_len,
    [[Specialize]] uint kernel_size,
    uint total_proj_dim,
    uint conv_dim,
    [[Specialize]] bool has_bias,
    [[Axis("suffix_len", "1")]] uint node_idx,
    [[Axis("total_proj_dim", "256")]] uint channel_idx,
) {
  uint3 gid = uint3(channel_idx, node_idx, 0);

  let channel = gid.x;
  let node = gid.y;
  if (channel >= total_proj_dim) {
    return;
  }
  let destination = node * total_proj_dim + channel;
  if (channel >= conv_dim) {
    tensorStore<T>(out_proj, destination, tensorLoad<T>(in_proj, destination));
    return;
  }
  let stateStride = kernel_size - 1u;
  var source = int(node);
  var sum = 0.0;
  if (has_bias) {
    sum = bias[channel];
  }
  for (var history = 0u; history < kernel_size; history++) {
    var sample = 0.0;
    if (source >= 0) {
      sample = tensorLoad<T>(in_proj, uint(source) * total_proj_dim + channel);
    } else {
      sample = base_state[channel * stateStride + stateStride - uint(-source)];
    }
    sum += sample * conv_weight[channel * kernel_size + kernel_size - 1u - history];
    if (history < stateStride) {
      suffix_state[(node * conv_dim + channel) * stateStride + stateStride - 1u - history] = sample;
    }
    if (source >= 0) {
      source = asint(parents[uint(source)]);
    } else {
      source--;
    }
  }
  tensorStore<T>(out_proj, destination, activate(sum, 0u, 1.0));
}
