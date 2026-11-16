import definitions;

[[Variants("T", "float, half, bf16")]]
[[Kernel]] [[Public]] func TensorAddScale<T: IStorageFloat>(
  [[Optional("!in_place")]] Ptr<T, Access.Read> input,
  [[Optional("has_bias")]] Ptr<T, Access.Read> bias,
  Ptr<T, Access.ReadWrite> output,
  uint num_cols,
  uint length,
  float scale,
  [[Axis("length", "32")]] uint position,
  [[Specialize]] bool in_place,
  [[Specialize]] bool has_bias,
) {
  let value = in_place_load(input, output, position, in_place);
  let bias_value = has_bias ? bias[position % num_cols].toFloat() : 0.0;
  output[position] = T.fromFloat((value + bias_value) * scale);
}
