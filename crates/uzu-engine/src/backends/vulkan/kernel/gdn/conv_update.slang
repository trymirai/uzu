import definitions;
import common.activation;
import generated.activation_type;

// One token in place: channel c accumulates in FP32 from the bias (or +0) over its state taps then the input, stores
// SiLU as T once, then shifts its state row down one tap and appends the input. The first precondition keeps the tap
// count from underflowing (kernel_size 0) and channel 0's last tap in bounds (kernel_size 1).

[[Kernel]] [[Public]]
[[Precondition("encode", "kernel_size >= 2 || kernel_size == 1 && conv_dim == 0")]]
[[Precondition("encode", "conv_dim <= 1 || state_stride >= kernel_size - 1")]]
func DeltaNetConvUpdate<T: IModelFloat>(
  Ptr<float, Access.Read> conv_weight,
  [[Optional("has_bias")]] Ptr<float, Access.Read> bias,
  Ptr<T, Access.ReadWrite> in_out,
  Ptr<float, Access.ReadWrite> state,
  uint kernel_size,
  uint conv_dim,
  uint state_stride,
  [[Axis("conv_dim", "64")]] uint channel,
  [[Specialize]] bool has_bias,
) {
  let taps = kernel_size - 1;
  let state_offset = channel * state_stride;
  let weight_row = channel * kernel_size;
  let x = in_out[channel].toFloat();
  var acc = 0.0;
  if (has_bias) {
    acc = bias[channel];
  }
  for (uint tap = 0; tap < taps; tap++) {
    acc = fma(state[state_offset + tap], conv_weight[weight_row + tap], acc);
  }
  acc = fma(x, conv_weight[weight_row + taps], acc);
  in_out[channel] = T.fromFloat(activate(acc, ActivationType.SILU));
  for (uint tap = 1; tap < taps; tap++) {
    state[state_offset + tap - 1] = state[state_offset + tap];
  }
  state[state_offset + taps - 1] = x;
}
