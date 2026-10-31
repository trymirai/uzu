import definitions;

// input and output [Q, C] of T, weights [C, K] and bias [C] of WeightT. Output (t, c) accumulates in FP32 from the bias
// (or +0) over taps j < min(K, t + 1) ascending of (weights[c, K - 1 - j] + coefficient_deltas[t R + j (C / G) + c / G])
// input[t - j, c], then rounds once. One invocation per output.

[[Variants("T", "bf16")]]
[[Kernel]] [[Public]]
[[Precondition("encode", "group_size > 0")]]
func SeparableCausalConv<T: IModelFloat16>(
  Ptr<T, Access.Read> input,
  Ptr<T, Access.Read> coefficient_deltas,
  Ptr<T, Access.Read> weights,
  [[Optional("has_bias")]] Ptr<T, Access.Read> bias,
  Ptr<T, Access.ReadWrite> output,
  uint sequence_length,
  uint coefficient_row_stride,
  [[Axis("model_dim", "64")]] uint channel,
  [[Axis("sequence_length", "1")]] uint token,
  [[Specialize]] uint model_dim,
  [[Specialize]] uint kernel_size,
  [[Specialize]] uint group_size,
  [[Specialize]] bool has_bias,
) {
  let num_groups = model_dim / group_size;
  let group = channel / group_size;
  var acc = 0.0;
  if (has_bias) {
    acc = bias[channel].toFloat();
  }
  let taps = min(kernel_size, token + 1);
  for (uint tap = 0; tap < taps; tap++) {
    let x = input[(token - tap) * model_dim + channel].toFloat();
    let w = weights[channel * kernel_size + (kernel_size - 1 - tap)].toFloat();
    let d = coefficient_deltas[token * coefficient_row_stride + tap * num_groups + group].toFloat();
    acc += (w + d) * x;
  }
  output[token * model_dim + channel] = T.fromFloat(acc);
}
