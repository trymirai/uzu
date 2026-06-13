import definitions;

[[Variants("T", "float, half")]]
[[Public]] [[Kernel]] func AudioConv1d<T: __BuiltinFloatingPointType>(
  StructuredBuffer<T> input,
  StructuredBuffer<T> weight,
  StructuredBuffer<T> bias,
  RWStructuredBuffer<T> output,
  StructuredBuffer<int> lengths,
  int cin,
  int cout,
  int seq_len_in,
  int seq_len_out,
  int kernel_size,
  int stride,
  int dilation,
  int padding,
  int pad_mode,
  int batch_size,
  [[Axis("seq_len_out", "32")]] uint t_out,
  [[Axis("cout", "1")]] uint oc,
  [[Axis("batch_size", "1")]] uint b,
) {
  // TODO
}