import definitions;
import common.matmul;
import matmul.gemv;
import matmul.gemv.common.arguments;
import matmul.gemv.common.epilogue;
import matmul.gemv.common.quant_b_source;
import matmul.gemv.common.reduce;
import matmul.gemv.common.tile;
import generated.quantization;
import generated.quantization_method;

// Metal non-MXU GEMV. Shapes outside its packed-row and group contract use the existing reader-based body.

[[Kernel]]
[[Precondition("new", "group_size > 0")]]
[[Precondition("new", "METAL_LAYOUT || (OUTPUT_ROW_TILE == 8 && NUM_SIMDGROUPS == 8 && INPUT_ROW_TILE == 1)")]]
[[Precondition("new", "quantization_mode == QuantizationMode::U4 || quantization_mode == QuantizationMode::U8")]]
[[Precondition("new", "!METAL_LAYOUT || (group_size == 16 || group_size == 32 || group_size == 64 || group_size == 128)")]]
[[Precondition("new", "!METAL_LAYOUT || ((1..=8).contains(&INPUT_ROW_TILE) && (INPUT_ROW_TILE != 1 || REDUCTION_LANES == 32))")]]
[[Precondition("new", "!METAL_LAYOUT || ([8, 16, 32].contains(&REDUCTION_LANES) && [1, 2, 4, 8, 16].contains(&GROUP_LANES) && REDUCTION_LANES.is_multiple_of(GROUP_LANES))")]]
[[Precondition("new", "!METAL_LAYOUT || ([1, 2, 4, 8, 16].contains(&GROUP_LANES) && group_size.is_multiple_of(GROUP_LANES) && (group_size / GROUP_LANES).is_multiple_of(if quantization_mode == QuantizationMode::U4 { 16 } else { 8 }))")]]
[[Precondition("new", "!METAL_LAYOUT || ([2, 4, 8].contains(&NUM_SIMDGROUPS) && OUTPUT_ROW_TILE.is_multiple_of(NUM_SIMDGROUPS) && [1, 2, 4, 8].contains(&(OUTPUT_ROW_TILE / NUM_SIMDGROUPS)))")]]
[[Precondition("new", "!METAL_LAYOUT || (![8, 16, 32].contains(&REDUCTION_LANES) || ![2, 4, 8].contains(&NUM_SIMDGROUPS) || (OUTPUT_ROW_TILE / NUM_SIMDGROUPS).is_multiple_of(32 / REDUCTION_LANES))")]]
[[Precondition("new", "!METAL_LAYOUT || (!gathered || INPUT_ROW_TILE == 1)")]]
[[Precondition("encode", "!METAL_LAYOUT || (quantization_mode == QuantizationMode::U8 || k.is_multiple_of(2))")]]
[[Precondition("encode", "!METAL_LAYOUT || (!INPUT_ALIGNED || k.is_multiple_of(group_size))")]]
[[Precondition("encode", "!METAL_LAYOUT || (!FULL_TILE || (n.is_multiple_of(OUTPUT_ROW_TILE) && m.is_multiple_of(INPUT_ROW_TILE)))")]]
func QuantizedGemv<A: IModelFloat, B: IModelFloat, D: IModelFloat>(
  Ptr<uint8_t, Access.Read> codes,
  Ptr<B, Access.Read> scales,
  [[Optional("quantization_method == QuantizationMethod::ScaleBias")]] Ptr<B, Access.Read> biases,
  [[Optional("quantization_method == QuantizationMethod::ScaleZeroPoint")]] Ptr<uint8_t, Access.Read> zero_points,
  Ptr<A, Access.Read> a,
  Ptr<D, Access.ReadWrite> d,
  [[Optional("has_bias")]] Ptr<B, Access.Read> bias,
  [[Optional("gathered")]] Ptr<uint, Access.Read> gather,
  uint k,
  uint n,
  uint m,
  uint scale_output_stride,
  uint scale_group_stride,
  [[Optional("quantization_method == QuantizationMethod::ScaleZeroPoint")]] uint zero_point_output_stride,
  [[Optional("quantization_method == QuantizationMethod::ScaleZeroPoint")]] uint zero_point_group_stride,
  [[Optional("has_scale")]] float ab_scale,
  [[Optional("has_soft_cap")]] float soft_cap,
  [[Groups("n.div_ceil(OUTPUT_ROW_TILE)")]] uint output_tile_idx,
  [[Groups("m.div_ceil(INPUT_ROW_TILE)")]] uint input_tile_idx,
  [[Threads("32")]] uint simd_lane,
  [[Threads("NUM_SIMDGROUPS")]] uint simd_group,
  [[Specialize]] bool has_scale,
  [[Specialize]] bool accumulate,
  [[Specialize]] bool has_bias,
  [[Specialize]] bool has_soft_cap,
  [[Specialize]] bool gathered,
  [[Specialize]] QuantizationMode quantization_mode,
  [[Specialize]] QuantizationMethod quantization_method,
  [[Specialize]] bool signed_codes,
  [[Specialize]] uint group_size,
  [[Specialize]] uint OUTPUT_ROW_TILE,
  [[Specialize]] uint NUM_SIMDGROUPS,
  [[Specialize]] uint INPUT_ROW_TILE,
  [[Specialize]] uint REDUCTION_LANES,
  [[Specialize]] uint GROUP_LANES,
  [[Specialize]] bool INPUT_ALIGNED,
  [[Specialize]] bool FULL_TILE,
  [[Specialize]] bool METAL_LAYOUT,
) {
  if (!METAL_LAYOUT) {
    QuantizedWeights<B> weights = { codes, scales, biases, zero_points, group_size, scale_output_stride,
      scale_group_stride, zero_point_output_stride, zero_point_group_stride, quantization_mode == QuantizationMode.U4,
      signed_codes, quantization_method };
    DenseOperand<A> activations = { a };
    gemv_body<DenseOperand<A>, B, D>(weights, activations, d, bias, gather, k, n, m, ab_scale, soft_cap,
      output_tile_idx, input_tile_idx, simd_group * 32 + simd_lane, has_scale, accumulate, has_bias, has_soft_cap, gathered);
    return;
  }
  let BITS = quantization_mode == QuantizationMode.U4 ? 4u : 8u;
  GemvOperands<A, B, D> ops = { gemv_cast<uint, uint8_t>(codes), scales, zero_points, biases, a, d, bias, gather };
  GemvParams params = {
    k, n, scale_output_stride, scale_group_stride, zero_point_output_stride, zero_point_group_stride, m, ab_scale,
    soft_cap, has_scale, accumulate, has_bias, has_soft_cap, gathered, signed_codes
  };
  let Tile = GemvTile(INPUT_ROW_TILE, OUTPUT_ROW_TILE, REDUCTION_LANES, GROUP_LANES, NUM_SIMDGROUPS, 1);
  let tile = OutputTile.make(Tile, FULL_TILE, output_tile_idx, input_tile_idx, simd_group, simd_lane, n);
  uint fields = 0x0000ffffu;
  float result[GEMV_MAX_INPUT_ROWS][GEMV_MAX_ROWS_PER_LANE];
  [unroll]
  for (uint I = 0; I < GEMV_MAX_INPUT_ROWS; I++) {
    [unroll]
    for (uint R = 0; R < GEMV_MAX_ROWS_PER_LANE; R++) {
      result[I][R] = 0.0;
    }
  }

  QuantBSource.accumulate(
    fields, result, ops, params, Tile, tile, quantization_method, group_size, BITS, INPUT_ALIGNED, FULL_TILE);
  Reduce.run(result, Tile, tile);
  fields = matmul_workgroup_fields(fields);
  QuantizedWeights<B> weights = {
    codes, scales, biases, zero_points, group_size, scale_output_stride, scale_group_stride,
    zero_point_output_stride, zero_point_group_stride, BITS == 4, signed_codes, quantization_method
  };
  DenseOperand<A> activations = { a };
  Epilogue.store(result, ops, params, Tile, tile, fields, FULL_TILE, activations, weights);
}
