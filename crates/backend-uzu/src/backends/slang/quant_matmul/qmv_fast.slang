import definitions;

static const uint METAL_SIMD_SIZE = 32;

[[Variants("T", "float, half")]]
[[Variants("GROUP_SIZE", "32, 64, 128")]]
[[Variants("BITS", "4, 8")]]
[[GroupShared("shared_results", "METAL_SIMD_SIZE")]]
[[Public]] [[Kernel]] func QuantizedMatmulQmvFast<T: __BuiltinFloatingPointType, let GROUP_SIZE: uint, let BITS: uint, shared_results: IGroupShared<float>>(
  StructuredBuffer<uint> weights,
  StructuredBuffer<T> scales,
  [[Optional("use_zero_points")]] StructuredBuffer<uint8_t> zero_points,
  [[Optional("use_mlx_quant")]] StructuredBuffer<T> biases,
  StructuredBuffer<T> input,
  RWStructuredBuffer<T> output,
  [[Optional("use_hadamard")]] StructuredBuffer<int> hadamard_factors,
  uint in_vec_size,
  uint out_vec_size,
  uint batch_size,
  [[Specialize]] bool use_zero_points,
  [[Specialize]] bool use_mlx_quant,
  [[Specialize]] bool use_hadamard,
  [[Groups("batch_size")]] uint batch_idx,
  [[Groups("out_vec_size.div_ceil(32)")]] uint out_block_idx,
  [[Threads("32")]] uint simd_lane,
  [[Threads("8")]] uint simd_group,
) {
  // TODO
}