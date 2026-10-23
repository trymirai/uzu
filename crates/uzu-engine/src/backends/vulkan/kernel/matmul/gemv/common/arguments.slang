import definitions;
import matmul.common.matmul;
import generated.gemm;
import generated.gemm_b_layout;
import generated.quantization;
import generated.quantization_method;

// b is a word pointer each B source reinterprets.

public struct GemvOperands<A : IModelFloat, B : IModelFloat, D : IModelFloat> {
  public Ptr<uint, Access.Read> b;
  public Ptr<B, Access.Read> scales;
  public Ptr<uint8_t, Access.Read> zero_points;
  public Ptr<B, Access.Read> biases;
  public Ptr<A, Access.Read> a;
  public Ptr<D, Access.ReadWrite> d;
  public Ptr<B, Access.Read> output_bias;
  public Ptr<uint, Access.Read> gather_indices;
}

public struct GemvParams {
  public uint in_vec_size;
  public uint out_vec_size;
  public uint scale_output_stride;
  public uint scale_group_stride;
  public uint zero_point_output_stride;
  public uint zero_point_group_stride;
  public uint batch_size;
  public float ab_scale;
  public float soft_cap;
  public bool has_scale;
  public bool has_bias;
  public bool has_soft_cap;
  public bool gathered;
  public bool signed_codes;
  public GemmBLayout b_layout;
}

// The quantization method of a B prologue; full precision has none and reads as symmetric.
public QuantizationMethod gemm_quantization_method(GemmBPrologueKind prologue) {
  switch (prologue) {
  case GemmBPrologueKind.ScaleBiasDequant:
    return QuantizationMethod.ScaleBias;
  case GemmBPrologueKind.ScaleZeroPointDequant:
    return QuantizationMethod.ScaleZeroPoint;
  default:
    return QuantizationMethod.ScaleSymmetric;
  }
}

public QuantizedWeights<B> gemm_weights<B : IModelFloat>(
  Ptr<uint, Access.Read> b,
  Ptr<B, Access.Read> scales,
  Ptr<B, Access.Read> biases,
  Ptr<uint8_t, Access.Read> zero_points,
  uint group_size,
  uint scale_output_stride,
  uint scale_group_stride,
  uint zero_point_output_stride,
  uint zero_point_group_stride,
  QuantizationMode quantization_mode,
  bool signed_codes,
  GemmBPrologueKind b_prologue,
  uint k,
  GemmBLayout b_layout,
) {
  QuantizedWeights<B> weights = {
    Ptr<uint8_t, Access.Read>(b), scales, biases, zero_points, group_size, scale_output_stride, scale_group_stride,
    zero_point_output_stride, zero_point_group_stride, quantization_mode == QuantizationMode.U4, signed_codes,
    gemm_quantization_method(b_prologue), k, b_layout
  };
  return weights;
}
