import definitions;
import common.matmul;
import matmul.gemm;
import generated.quantization;
import generated.quantization_method;

// A8QuantizedGemm: A8QuantizedGemv's INT8 A and U4/U8 B through matmul.gemm's body, without gather, in its own module.
// A8 K is a multiple of a_group_size, hence of GEMM_DEPTH.

[[Kernel]]
[[Precondition("new", "group_size == 32 || group_size == 64 || group_size == 128")]]
[[Precondition("new", "a_group_size == 32 || a_group_size == 64 || a_group_size == 128")]]
[[Precondition("new", "quantization_mode == QuantizationMode::U4 || quantization_mode == QuantizationMode::U8")]]
[[Precondition("encode", "k.is_multiple_of(a_group_size)")]]
func A8QuantizedGemm<B: IModelFloat, D: IModelFloat>(
  Ptr<uint8_t, Access.Read> codes,
  Ptr<B, Access.Read> scales,
  [[Optional("quantization_method == QuantizationMethod::ScaleBias")]] Ptr<B, Access.Read> biases,
  [[Optional("quantization_method == QuantizationMethod::ScaleZeroPoint")]] Ptr<uint8_t, Access.Read> zero_points,
  Ptr<int8_t, Access.Read> a_codes,
  Ptr<float, Access.Read> a_scales,
  Ptr<D, Access.ReadWrite> d,
  [[Optional("has_bias")]] Ptr<B, Access.Read> bias,
  uint k,
  uint n,
  uint m,
  uint scale_output_stride,
  uint scale_group_stride,
  [[Optional("quantization_method == QuantizationMethod::ScaleZeroPoint")]] uint zero_point_output_stride,
  [[Optional("quantization_method == QuantizationMethod::ScaleZeroPoint")]] uint zero_point_group_stride,
  [[Optional("has_scale")]] float ab_scale,
  [[Optional("has_soft_cap")]] float soft_cap,
  [[Groups("n.div_ceil(64)")]] uint column_tile,
  [[Groups("m.div_ceil(64)")]] uint row_tile,
  [[Threads("256")]] uint tid,
  [[Specialize]] bool has_scale,
  [[Specialize]] bool accumulate,
  [[Specialize]] bool has_bias,
  [[Specialize]] bool has_soft_cap,
  [[Specialize]] QuantizationMode quantization_mode,
  [[Specialize]] QuantizationMethod quantization_method,
  [[Specialize]] bool signed_codes,
  [[Specialize]] uint group_size,
  [[Specialize]] uint a_group_size,
  [[Specialize]] bool grouped_by_weight_nibble,
) {
  QuantizedWeights<B> weights = {
    codes, scales, biases, zero_points, group_size, scale_output_stride, scale_group_stride,
    zero_point_output_stride, zero_point_group_stride, quantization_mode == QuantizationMode.U4, signed_codes,
    quantization_method
  };
  Int8Activations activations = { a_codes, a_scales, a_group_size, grouped_by_weight_nibble };
  gemm_body<Int8Activations, B, D>(
    weights, activations, d, bias, k, n, m, ab_scale, soft_cap, column_tile, row_tile, tid, has_scale, accumulate,
    has_bias, has_soft_cap);
}
