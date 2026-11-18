import definitions;
import common.activation;
import common.quantized_value;
import generated.gemm;
import generated.gemm_b_layout;
import generated.quantization;
import generated.quantization_method;

public interface IMatmulOperand {
  public float value(uint row, uint inner);
}

public struct DenseOperand<T : IStorageFloat> : IMatmulOperand {
  public Ptr<T, Access.Read> values;
  public uint k;

  public float value(uint row, uint inner) {
    return values[row * k + inner].toFloat();
  }
}

public struct WeightGroup {
  public float scale;
  public float bias;
  public uint zero_point;
}

public struct QuantizedWeights<B : IStorageFloat> : IMatmulOperand {
  public Ptr<uint8_t, Access.Read> codes;
  public Ptr<B, Access.Read> scales;
  public Ptr<B, Access.Read> biases;
  public Ptr<uint8_t, Access.Read> zero_points;
  public uint group_size;
  public uint scale_output_stride;
  public uint scale_group_stride;
  public uint zero_point_output_stride;
  public uint zero_point_group_stride;
  public bool u4;
  public bool signed_codes;
  public QuantizationMethod method;
  public uint k;
  public GemmBLayout layout;

  public uint code_byte(uint row, uint inner) {
    let shift = u4 ? 1u : 0u;
    if (layout == GemmBLayout.Tiled) {
      let tile = row / 8 * (k / 64) + inner / 64;
      return tile * (512u >> shift) + (((row % 8) * 64 + inner % 64) >> shift);
    }
    return (row * k + inner) >> shift;
  }

  public WeightGroup metadata(uint row, uint group) {
    let index = row * scale_output_stride + group * scale_group_stride;
    WeightGroup metadata;
    metadata.scale = scales[index].toFloat();
    metadata.bias = 0.0;
    metadata.zero_point = u4 ? 8u : 128u;
    if (method == QuantizationMethod.ScaleBias) {
      metadata.bias = biases[index].toFloat();
      metadata.zero_point = 0;
    } else if (method == QuantizationMethod.ScaleZeroPoint) {
      let offset = row * zero_point_output_stride + group * zero_point_group_stride;
      metadata.zero_point = u4 ? read_nibble(zero_points, offset) : uint(zero_points[offset]);
    }
    return metadata;
  }

  public float value(uint row, uint inner) {
    let byte = uint(codes[code_byte(row, inner)]);
    let nibble = layout == GemmBLayout.RowMajor ? (row * k + inner) & 1u : inner & 1u;
    var code = u4 ? (byte >> (4 * nibble)) & 15u : byte;
    if (signed_codes) {
      code ^= u4 ? 8u : 128u;
    }
    let group = metadata(row, inner / group_size);
    let scale_bias = method == QuantizationMethod.ScaleBias;
    return quantized_value(group.scale, int(code), scale_bias, scale_bias ? group.bias : float(group.zero_point));
  }
}

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

public QuantizedWeights<B> gemm_weights<B : IStorageFloat>(
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

public struct GemmParams {
  public uint M;
  public uint N;
  public uint K;
  public float ab_scale;
  public float soft_cap;
  public bool has_scale;
  public bool has_bias;
  public bool has_soft_cap;
  public bool gathered;
  public uint alignment;
  public uint d_row_stride;
  public uint gather_first_row;
  public uint gather_row_count;

  // A gathered output reads weight row index - gather_first_row; indices outside the window are not written.
  public bool gathers(uint index) {
    return index - gather_first_row < gather_row_count;
  }

  public uint gathered_row(uint index) {
    return gathers(index) ? index - gather_first_row : 0;
  }
}

public float gemm_output(float dot, float bias, GemmParams params) {
  var value = params.has_scale ? dot * params.ab_scale : dot;
  if (params.has_bias) {
    value += bias;
  }
  if (params.has_soft_cap) {
    value = apply_soft_cap(value, params.soft_cap);
  }
  return value;
}
