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
  T value;
  if (in_place) {
    value = output[position];
  } else {
    value = input[position];
  }
  output[position] = T.fromFloat(value.toFloat() + bias[position % num_cols].toFloat());
}
