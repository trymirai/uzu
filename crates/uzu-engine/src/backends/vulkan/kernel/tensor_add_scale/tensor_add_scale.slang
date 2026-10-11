import definitions;

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
  T value;
  if (in_place) {
    value = output[position];
  } else {
    value = input[position];
  }
  output[position] = T.fromFloat((value.toFloat() + bias[position % num_cols].toFloat()) * scale);
}
