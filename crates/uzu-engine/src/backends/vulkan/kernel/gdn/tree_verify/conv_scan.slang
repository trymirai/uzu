import definitions;
import common.activation;
import generated.activation_type;

// Channel c < conv_dim of node `row` accumulates in FP32 from the bias (or +0) over history taps ascending, each
// sample found by walking parents and then base_state, and stores T(silu(acc)); the first K - 1 samples are recorded
// raw in suffix_state. Channels from conv_dim on copy their T bits. out_proj must not alias in_proj.

[[Kernel]] [[Public]]
[[Precondition("new", "kernel_size >= 1")]]
func ConvTreeScan<T: IModelFloat>(
  Ptr<T, Access.Read> in_proj,
  Ptr<float, Access.Read> conv_weight,
  [[Optional("has_bias")]] Ptr<float, Access.Read> bias,
  Ptr<float, Access.Read> base_state,
  Ptr<int, Access.Read> parents,
  Ptr<T, Access.ReadWrite> out_proj,
  Ptr<float, Access.ReadWrite> suffix_state,
  uint suffix_len,
  uint total_proj_dim,
  uint conv_dim,
  [[Axis("total_proj_dim", "64")]] uint channel,
  [[Axis("suffix_len", "1")]] uint row,
  [[Specialize]] uint kernel_size,
  [[Specialize]] bool has_bias,
) {
  let proj = row * total_proj_dim + channel;
  if (channel >= conv_dim) {
    out_proj[proj] = in_proj[proj];
    return;
  }
  let taps = kernel_size - 1;
  var acc = 0.0;
  if (has_bias) {
    acc = bias[channel];
  }
  let state_offset = (row * conv_dim + channel) * taps;
  var source = int(row);
  for (uint h = 0; h < kernel_size; h++) {
    var sample = 0.0;
    if (source >= 0) {
      sample = in_proj[uint(source) * total_proj_dim + channel].toFloat();
    } else {
      sample = base_state[channel * taps + taps - uint(-source)];
    }
    let weight = conv_weight[channel * kernel_size + taps - h];
    acc = fma(sample, weight, acc);
    if (h < taps) {
      suffix_state[state_offset + taps - 1 - h] = sample;
      source = source >= 0 ? parents[uint(source)] : source - 1;
    }
  }
  out_proj[proj] = T.fromFloat(activate(acc, ActivationType.SILU));
}
