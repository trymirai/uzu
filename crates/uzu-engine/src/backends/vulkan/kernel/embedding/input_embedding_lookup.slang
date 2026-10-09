import definitions;
import common.activation;
import common.hadamard;
import common.quantized_value;
import generated.d4s4;
import generated.embedding;
import generated.hadamard_order;
import generated.quantization;
import generated.quantization_method;

static const uint INPUT_EMBEDDING_LOOKUP_TILE = 256;

// One row per token, one workgroup per tile of a row, in the CPU's staging with exact FP32 products: a dense T value
// times the input scale rounded to T is rounded to T; a quantized code times its group scale plus the group bias (the
// scale times minus the zero point or the midpoint, 8 for U4 and 128 otherwise, also I8), times the input scale, is
// rounded to T; D4S4 multiplies the row scale, ladder step, codebook point and input scale in FP32. The output random
// Hadamard transform runs in FP32 before the factors and the final rounding to T. A token outside the vocabulary gives
// positive zeros without reading any table; it is uniform over the workgroup, which returns before the transform's
// barriers, while for valid tokens every invocation reaches them: tiles past the row hold neutral zeros.
[[Kernel]] [[Public]]
[[Precondition("new", "group_size.unwrap_or(1) > 0")]]
[[Precondition("encode", "quantization_mode != QuantizationMode::U4.into() || model_dim.is_multiple_of(2)")]]
[[Precondition("encode", "!use_hadamard || model_dim.is_multiple_of(32)")]]
[[Precondition("encode", "table_kind != EmbeddingTableKind::D4S4 || model_dim.is_multiple_of(64)")]]
func InputEmbeddingLookup<T: IModelFloat>(
  Ptr<uint, Access.Read> token_ids,
  Ptr<uint8_t, Access.Read> values,
  [[Optional("table_kind != EmbeddingTableKind::Dense")]] Ptr<T, Access.Read> scales,
  [[Optional("quantization_method == Some(QuantizationMethod::ScaleZeroPoint)")]] Ptr<uint8_t, Access.Read> zero_points,
  [[Optional("quantization_method == Some(QuantizationMethod::ScaleBias)")]] Ptr<T, Access.Read> biases,
  [[Optional("use_hadamard")]] Ptr<int, Access.Read> hadamard_factors,
  [[Optional("table_kind == EmbeddingTableKind::D4S4")]] Ptr<uint8_t, Access.Read> ladder_indices,
  [[Optional("table_kind == EmbeddingTableKind::D4S4")]] Ptr<half, Access.Read> ladder,
  [[Optional("table_kind == EmbeddingTableKind::D4S4")]] Ptr<int8_t, Access.Read> codebook,
  Ptr<T, Access.ReadWrite> output,
  uint batch_size,
  uint vocab_size,
  uint model_dim,
  float input_scale,
  [[Groups("model_dim.div_ceil(256)")]] uint tile,
  [[Groups("batch_size")]] uint batch,
  [[Threads("256")]] uint tid,
  [[Specialize]] EmbeddingTableKind table_kind,
  [[Specialize]] [[Optional("table_kind == EmbeddingTableKind::Quantized")]] uint group_size,
  [[Specialize]] [[Optional("table_kind == EmbeddingTableKind::Quantized")]] QuantizationMode quantization_mode,
  [[Specialize]] [[Optional("table_kind == EmbeddingTableKind::Quantized")]] QuantizationMethod quantization_method,
  [[Specialize]] bool use_hadamard,
) {
  static_assert(INPUT_EMBEDDING_LOOKUP_TILE == 256, "one tile of Threads(256) per workgroup, Groups(div_ceil(256))");
  static_assert(HADAMARD_TRANSFORM_BLOCK_SIZE == 32, "the transform precondition admits rows of multiples of 32");
  static_assert(COLUMNS_PER_LADDER_SCALE == 64, "the D4S4 precondition admits rows of multiples of 64");
  static_assert(COLUMNS_PER_LADDER_SCALE % VALUES_PER_CODE == 0, "D4S4 rows hold whole codes");
  static_assert(CODEBOOK_SIZE == 256 && LADDER_SIZE == 16, "a code byte indexes the codebook, a nibble the ladder");
  static_assert(COLUMNS_PER_LADDER_INDEX_BYTE == 2 * COLUMNS_PER_LADDER_SCALE, "two ladder nibbles per byte");
  let column = tile * INPUT_EMBEDDING_LOOKUP_TILE + tid;
  let active = column < model_dim;
  let index = batch * model_dim + column;
  let token = token_ids[batch];
  if (token >= vocab_size) {
    if (active) {
      output[index] = T.fromFloat(0.0);
    }
    return;
  }
  float value = 0.0;
  if (active) {
    if (table_kind == EmbeddingTableKind.Dense) {
      let table = Ptr<T, Access.Read>(values);
      let scale = T.fromFloat(input_scale).toFloat();
      value = T.fromFloat(exact_product(table[token * model_dim + column].toFloat(), scale)).toFloat();
    } else if (table_kind == EmbeddingTableKind.Quantized) {
      let u4 = quantization_mode == QuantizationMode.U4;
      let group = column / group_size;
      let groups = (model_dim - 1) / group_size + 1;
      let scale_index = token * groups + group;
      let scale = scales[scale_index].toFloat();
      let row = token * (u4 ? model_dim / 2 : model_dim);
      int code;
      if (u4) {
        code = int(read_nibble(values, 2 * row + column));
      } else if (quantization_mode == QuantizationMode.I8) {
        code = int(int8_t(values[row + column]));
      } else {
        code = int(uint(values[row + column]));
      }
      let scale_bias = quantization_method == QuantizationMethod.ScaleBias;
      float correction;
      if (scale_bias) {
        correction = biases[scale_index].toFloat();
      } else {
        correction = quantization_method == QuantizationMethod.ScaleZeroPoint
                       ? float(u4 ? read_nibble(zero_points, 2 * token * ((groups + 1) / 2) + group)
                                  : uint(zero_points[scale_index]))
                       : (u4 ? 8.0 : 128.0);
      }
      value = T.fromFloat(exact_product(quantized_value(scale, code, scale_bias, correction), input_scale)).toFloat();
    } else {
      let ladder_index =
        read_nibble(ladder_indices, token * (model_dim / COLUMNS_PER_LADDER_SCALE) + column / COLUMNS_PER_LADDER_SCALE);
      let code = uint(values[token * (model_dim / VALUES_PER_CODE) + column / VALUES_PER_CODE]);
      let point = float(int(codebook[VALUES_PER_CODE * code + column % VALUES_PER_CODE]));
      let step = exact_product(scales[token].toFloat(), ladder[ladder_index].toFloat());
      value = exact_product(exact_product(step, point), input_scale);
    }
  }
  if (use_hadamard) {
    value = hadamard_transform<INPUT_EMBEDDING_LOOKUP_TILE>(value, tid);
    if (active) {
      value = hadamard_factor(value, hadamard_factors[column]);
    }
  }
  if (active) {
    output[index] = T.fromFloat(value);
  }
}
