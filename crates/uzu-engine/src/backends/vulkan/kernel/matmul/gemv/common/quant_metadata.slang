import definitions;
import common.quantized_value;
import generated.quantization_method;
import matmul.gemv.common.arguments;
import matmul.gemv.common.quant_chunk;
import matmul.gemv.common.tile;

public struct QuantMetadata {
  float scale[GEMV_MAX_ROWS_PER_LANE];
  float origin[GEMV_MAX_ROWS_PER_LANE];
  float bias[GEMV_MAX_ROWS_PER_LANE];
  bool signed_codes;
  QuantizationMethod B_PROLOGUE;
  uint BITS;

  public static uint group_count(GemvParams params, uint group_size) {
    return (params.in_vec_size + group_size - 1) / group_size;
  }

  [mutating]
  public void load<A : IModelFloat, B : IModelFloat, D : IModelFloat>(
    uint group_index,
    uint weight_rows[GEMV_MAX_ROWS_PER_LANE],
    GemvOperands<A, B, D> ops,
    GemvParams params,
    GemvTile Tile,
    QuantizationMethod b_prologue,
    uint bits
  ) {
    B_PROLOGUE = b_prologue;
    BITS = bits;
    signed_codes = params.signed_codes;
    [unroll]
    for (uint R = 0; R < Tile.ROWS_PER_LANE; R++) {
      let row = weight_rows[R];
      let scale_index = row * params.scale_output_stride + group_index * params.scale_group_stride;
      scale[R] = ops.scales[scale_index].toFloat();
      if (B_PROLOGUE == QuantizationMethod.ScaleZeroPoint) {
        let zero_point_index = row * params.zero_point_output_stride + group_index * params.zero_point_group_stride;
        let zero_point = BITS == 4 ? read_nibble(ops.zero_points, zero_point_index) : uint(ops.zero_points[zero_point_index]);
        origin[R] = QuantChunk.MANTISSA_BASE + float(zero_point);
        bias[R] = 0.0;
      } else if (B_PROLOGUE != QuantizationMethod.ScaleBias) {
        origin[R] = QuantChunk.MANTISSA_BASE + float(1u << (BITS - 1));
        bias[R] = 0.0;
      } else {
        origin[R] = QuantChunk.MANTISSA_BASE;
        bias[R] = ops.biases[scale_index].toFloat();
      }
    }
  }

  public void decode(uint4 words[GEMV_MAX_SLICE_WORDS], uint chunk, uint row, out float values[QuantChunk.VALUES]) {
    QuantChunk.decode(words, chunk, values, origin[row], signed_codes, BITS);
  }

  public void fold(
    inout float result[GEMV_MAX_INPUT_ROWS][GEMV_MAX_ROWS_PER_LANE],
    float partial[GEMV_MAX_INPUT_ROWS][GEMV_MAX_ROWS_PER_LANE],
    float input_sum[GEMV_MAX_INPUT_ROWS],
    GemvTile Tile
  ) {
    [unroll]
    for (uint I = 0; I < Tile.INPUT_ROWS; I++) {
      [unroll]
      for (uint R = 0; R < Tile.ROWS_PER_LANE; R++) {
        result[I][R] = finish(result[I][R], partial[I][R], input_sum[I], R);
      }
    }
  }

  float finish(float prior, float partial, float input_sum, uint row) {
    var result = fma(scale[row], partial, prior);
    if (B_PROLOGUE == QuantizationMethod.ScaleBias) {
      result = fma(bias[row], input_sum, result);
    }
    return result;
  }
}
