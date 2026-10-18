import definitions;
import storage;
import generated.embedding;
import generated.quantization;
import generated.quantization_method;

groupshared float h[256];
func code4(StructuredBuffer<uint> values, index : uint) -> float {
  let raw = uint(byteLoad(values, index / 2u));
  return float((raw >> ((index % 2u) * 4u)) & 15u);
}

[[Variants("T", "float, half, BFloat16")]]
[[Public]] [[Kernel]] func InputEmbeddingLookup<T : ITensorElement>(
    StructuredBuffer<uint> token_ids,
    StructuredBuffer<uint> values,
    [[Optional("table_kind != EmbeddingTableKind::Dense")]] StructuredBuffer<uint> scales,
    [[Optional(
        "table_kind == EmbeddingTableKind::Quantized && quantization_method == QuantizationMethod::ScaleZeroPoint"
    )]] StructuredBuffer<uint> zero_points,
    [[Optional(
        "table_kind == EmbeddingTableKind::Quantized && quantization_method == QuantizationMethod::ScaleBias"
    )]] StructuredBuffer<uint> biases,
    [[Optional("use_hadamard")]] StructuredBuffer<int> hadamard_factors,
    [[Optional("table_kind == EmbeddingTableKind::D4S4")]] StructuredBuffer<uint> ladder_indices,
    [[Optional("table_kind == EmbeddingTableKind::D4S4")]] StructuredBuffer<half> ladder,
    [[Optional("table_kind == EmbeddingTableKind::D4S4")]] StructuredBuffer<vector<int8_t, 4>> codebook,
    RWStructuredBuffer<Atomic<uint>> output,
    uint batch_size,
    uint vocab_size,
    uint model_dim,
    float input_scale,
    [[Specialize]] EmbeddingTableKind table_kind,
    [[Specialize]] [[Optional("table_kind == EmbeddingTableKind::Quantized")]] uint group_size,
    [[Specialize]] [[Optional("table_kind == EmbeddingTableKind::Quantized")]] QuantizationMode quantization_mode,
    [[Specialize]] [[Optional("table_kind == EmbeddingTableKind::Quantized")]] QuantizationMethod quantization_method,
    [[Specialize]] bool use_hadamard,
    [[Groups("model_dim.div_ceil(256)")]] uint dim_group,
    [[Threads("256")]] uint lane,
    [[Groups("batch_size")]] uint batch_idx,
) {
  uint3 gid = uint3(dim_group, batch_idx, 0);

  let col = gid.x * 256u + lane;
  let batch = gid.y;
  let token = token_ids[batch];
  var v = 0.0;
  if (col < model_dim && token < vocab_size) {
    if (table_kind == EmbeddingTableKind::Dense) {
      v = tensorRound<T>(tensorLoad<T>(values, token * model_dim + col) * tensorRound<T>(input_scale));
    }
    if (table_kind == EmbeddingTableKind::Quantized) {
      let groups = (model_dim + group_size - 1u) / group_size;
      let index = token * groups + col / group_size;
      let scale = tensorLoad<T>(scales, index);
      var code = 0.0;
      if (quantization_mode == QuantizationMode::U4) {
        code = code4(values, token * model_dim + col);
      } else {
        let raw = uint(byteLoad(values, token * model_dim + col));
        code = float(raw);
        if (quantization_mode == QuantizationMode::I8) {
          code = float(asint(raw << 24u) >> 24);
        }
      }
      var bias = 0.0;
      if (quantization_method == QuantizationMethod::ScaleBias) {
        bias = tensorLoad<T>(biases, index);
      }
      if (quantization_method == QuantizationMethod::ScaleZeroPoint) {
        var zero = float(byteLoad(zero_points, index));
        if (quantization_mode == QuantizationMode::U4) {
          let nibble = token * ((groups + 1u) / 2u) * 2u + col / group_size;
          zero = float((uint(byteLoad(zero_points, nibble / 2u)) >> ((nibble % 2u) * 4u)) & 15u);
        }
        bias = -scale * zero;
      }
      if (quantization_method == QuantizationMethod::ScaleSymmetric) {
        bias = -scale * choose(128.0, 8.0, quantization_mode == QuantizationMode::U4);
      }
      v = tensorRound<T>((scale * code + bias) * input_scale);
    }
  }
  if (use_hadamard) {
    h[lane] = v;
    GroupMemoryBarrierWithGroupSync();
    for (var stride = 1u; stride < 32u; stride *= 2u) {
      let own = h[lane];
      let paired = h[lane ^ stride];
      GroupMemoryBarrierWithGroupSync();
      h[lane] = choose(own + paired, paired - own, (lane & stride) != 0u);
      GroupMemoryBarrierWithGroupSync();
    }
    if (col < model_dim)
      v = h[lane] * 0.1767766952966369 * hadamard_factors[col];
  }
  if (col < model_dim) {
    tensorStore<T>(output, batch * model_dim + col, v);
  }
}
