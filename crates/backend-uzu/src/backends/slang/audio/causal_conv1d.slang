import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func AudioCausalConv1d<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> input,
  StructuredBuffer<T> weight,
  StructuredBuffer<T> bias,
  RWStructuredBuffer<T> output,
  StructuredBuffer<int> lengths,
  int cin,
  int cout,
  int seq_len,
  int kernel_size,
  int dilation,
  int input_layout,
  int batch_size,
  [[Axis("(seq_len + 3) / 4", "32")]] uint t,
  [[Axis("cout", "1")]] uint oc,
  [[Axis("batch_size", "1")]] uint b,
) {
  // TODO
}