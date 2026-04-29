import definitions;

[[Variants("T", "float, half")]]
[[Variants("GROUP_SIZE", "32, 64, 128")]]
[[Variants("BITS", "4, 8")]]
[[Public]] [[Kernel]] func QuantizedMatmulQmv<T: __BuiltinFloatingPointType, let GROUP_SIZE: uint, let BITS: uint>(
  StructuredBuffer<uint> weights,
  StructuredBuffer<T> scales,
  [[Optional("use_zero_points")]] StructuredBuffer<uint8_t> zero_points,
  [[Optional("use_mlx_quant")]] StructuredBuffer<T> biases,
  StructuredBuffer<T> input,
  RWStructuredBuffer<T> output,
  uint in_vec_size,
  uint out_vec_size,
  uint batch_size,
  [[Specialize]] bool use_zero_points,
  [[Specialize]] bool use_mlx_quant,
  [[Groups("batch_size")]] uint batch_idx,
  [[Groups("out_vec_size.div_ceil(8)")]] uint out_block_idx,
  [[Threads("32")]] uint simd_lane,
  [[Threads("2")]] uint simd_group,
) {
  // TODO
}