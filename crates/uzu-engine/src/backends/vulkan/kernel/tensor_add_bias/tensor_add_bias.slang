import definitions;

[[Kernel]] [[Public]] func TensorAddBias<T: IStorageFloat, BiasT: IStorageFloat>(
  [[Optional("!in_place")]] Ptr<T, Access.Read> input,
  Ptr<BiasT, Access.Read> bias,
  Ptr<T, Access.ReadWrite> output,
  uint num_cols,
  uint length,
  [[Axis("length", "32")]] uint position,
  [[Specialize]] bool in_place,
) {
  let value = in_place_load(input, output, position, in_place);
  output[position] = T.fromFloat(value + bias[position % num_cols].toFloat());
}
