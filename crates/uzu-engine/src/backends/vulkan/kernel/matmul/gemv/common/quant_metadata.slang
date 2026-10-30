import definitions;
import matmul.common.matmul;
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
  public void load<B : IModelFloat>(
    uint group_index,
    uint weight_rows[GEMV_MAX_ROWS_PER_LANE],
    QuantizedWeights<B> weights,
    GemvTile Tile
  ) {
    B_PROLOGUE = weights.method;
    BITS = weights.u4 ? 4 : 8;
    signed_codes = weights.signed_codes;
    [unroll]
    for (uint R = 0; R < Tile.ROWS_PER_LANE; R++) {
      let group = weights.metadata(weight_rows[R], group_index);
      scale[R] = group.scale;
      origin[R] = QuantChunk.MANTISSA_BASE + float(group.zero_point);
      bias[R] = group.bias;
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
