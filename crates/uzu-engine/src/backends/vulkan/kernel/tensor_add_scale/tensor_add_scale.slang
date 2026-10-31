import definitions;

[[Variants("T", "float, half, bf16")]]
[[Kernel]] [[Public]] func TensorAddScale<T: IStorageFloat>(
  [[Optional("!in_place")]] Ptr<T, Access.Read> input,
  Ptr<T, Access.Read> bias,
  Ptr<T, Access.ReadWrite> output,
  uint num_cols,
  uint length,
  float scale,
  [[Axis("length", "32")]] uint position,
  [[Specialize]] bool in_place,
) {
  let value = in_place_load(input, output, position, in_place);
  output[position] = T.fromFloat((value + bias[position % num_cols].toFloat()) * scale);
}
