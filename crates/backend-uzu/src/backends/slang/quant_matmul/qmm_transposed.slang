import definitions;

[[Variants("T", "float, half")]]
[[Variants("GROUP_SIZE", "32, 64, 128")]]
[[Variants("BITS", "4, 8")]]
[[Variants("BM", "8, 32, 64")]]
[[Variants("BK", "32, 64")]]
[[Variants("BN", "32, 64")]]
[[Variants("WM", "1, 2")]]
[[Variants("WN", "1, 2")]]
[[GroupShared("Xs", "BM * (BK + 16 / sizeof(T))")]]
[[GroupShared("Ws", "BN * (BK + 16 / sizeof(T))")]]
[[Constraint("(BM == 8 && BK == 32 && BN == 32 && WM == 1 && WN == 1) || (BM == 32 && BK == 32 && BN == 32 && WM == 2 && WN == 2) || (BM == 32 && BK == 64 && BN == 32 && WM == 2 && WN == 2) || (BM == 64 && BK == 32 && BN == 64 && WM == 2 && WN == 2) || (BM == 64 && BK == 64 && BN == 64 && WM == 2 && WN == 2)")]]
[[Constraint("BK <= GROUP_SIZE")]]
[[Constraint("T != \"float\" || BK < 64")]]
[[Public]] [[Kernel]] func QuantizedMatmulQmmTransposed<T: __BuiltinFloatingPointType, let GROUP_SIZE: uint, let BITS: uint, let BM: uint, let BK: uint, let BN: uint, let WM: uint, let WN: uint, Xs: IGroupShared<T>, Ws: IGroupShared<T>>(
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
  [[Specialize]] bool aligned_n,
  [[Groups("out_vec_size.div_ceil(BN)")]] uint out_block_idx,
  [[Groups("batch_size.div_ceil(BM)")]] uint batch_block_idx,
  [[Threads("32")]] uint simd_lane,
  [[Threads("WM * WN")]] uint simd_group,
) {
  // TODO
}