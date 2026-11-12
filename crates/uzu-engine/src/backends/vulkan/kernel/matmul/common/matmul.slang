import definitions;
import common.activation;
import common.quantized_value;
import generated.gemm_b_layout;
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

public uint gemm_b_code_byte(GemmBLayout layout, uint row, uint inner, uint k, bool u4) {
  let shift = u4 ? 1u : 0u;
  if (layout == GemmBLayout.Tiled) {
    let tile = row / 8 * (k / 64) + inner / 64;
    return tile * (512u >> shift) + (((row % 8) * 64 + inner % 64) >> shift);
  }
  return (row * k + inner) >> shift;
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
    return gemm_b_code_byte(layout, row, inner, k, u4);
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

public float matmul_epilogue(
  float dot,
  bool has_scale,
  float ab_scale,
  bool has_bias,
  float bias,
  bool has_soft_cap,
  float soft_cap
) {
  var value = has_scale ? dot * ab_scale : dot;
  if (has_bias) {
    value += bias;
  }
  if (has_soft_cap) {
    value = apply_soft_cap(value, soft_cap);
  }
  return value;
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
  public uint alignment;
}

public float gemm_output(float dot, float bias, GemmParams params) {
  return matmul_epilogue(
    dot, params.has_scale, params.ab_scale, params.has_bias, bias, params.has_soft_cap, params.soft_cap);
}

public interface IThreadgroupStorage<T> {
  public static T read(uint index);
  public static void write(uint index, T value);
}
