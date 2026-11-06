import definitions;
import storage;
import attention.common;
import generated.qkv_norm;

[[Variants("ElementT", "BFloat16, float")]]
[[Variants("RopeT", "float")]]
[[Variants("HEAD_DIM", "64, 128, 256, 512")]]
[[Public]] [[Kernel]] func AttentionPrepare<ElementT : ITensorElement, RopeT : ITensorElement, let HEAD_DIM: uint32_t>(
  StructuredBuffer<uint32_t> qkvg,
  RWStructuredBuffer<Atomic<uint32_t>> q_codes,
  RWStructuredBuffer<Atomic<uint32_t>> q_scales,
  [[Optional("append_kv")]] RWStructuredBuffer<Atomic<uint32_t>> k_codes,
  [[Optional("append_kv")]] RWStructuredBuffer<Atomic<uint32_t>> k_scales,
  [[Optional("append_kv")]] RWStructuredBuffer<Atomic<uint32_t>> v_codes,
  [[Optional("append_kv")]] RWStructuredBuffer<Atomic<uint32_t>> v_scales,
  [[Optional("has_rope")]] StructuredBuffer<uint32_t> cosines,
  [[Optional("has_rope")]] StructuredBuffer<uint32_t> sines,
  [[Optional("has_query_norm_scales")]] StructuredBuffer<float> query_norm_scales,
  [[Optional("has_key_norm_scales")]] StructuredBuffer<float> key_norm_scales,
  [[Optional("has_value_norm_scales")]] StructuredBuffer<float> value_norm_scales,
  QKVNormParams norm_params,
  uint32_t num_q_heads,
  [[Optional("has_rope")]] uint32_t rope_dim,
  uint32_t kv_token_offset,
  uint32_t input_row_stride,
  uint32_t suffix_length,
  uint32_t query_rows,
  [[Specialize]] bool has_rope,
  [[Specialize]] uint32_t num_kv_heads,
  [[Specialize]] uint32_t backend_opaque_layout,
  [[Specialize]] bool append_kv,
  [[Specialize]] bool has_query_norm_scales,
  [[Specialize]] bool has_key_norm_scales,
  [[Specialize]] bool has_value_norm_scales,
  [[Groups("suffix_length")]] uint32_t token_idx,
  [[Groups("num_q_heads + num_kv_heads * 2")]] uint32_t head_idx,
  [[Threads("1")]] uint32_t lane,
) {
  // One invocation per head and token: intentionally simple reference kernels.
  bool is_query = head_idx < num_q_heads;
  if (!append_kv && !is_query)
    return;
  bool is_key = !is_query && head_idx < num_q_heads + num_kv_heads;
  uint32_t norm = is_query ? 0 : (is_key ? 1 : 2);
  uint32_t base = token_idx * input_row_stride + head_idx * HEAD_DIM;
  float values[HEAD_DIM];
  float normalized[HEAD_DIM];
  float sum = 0.0;
  for (uint32_t channel = 0; channel < HEAD_DIM; ++channel) {
    float value = tensorLoad<ElementT>(qkvg, base + channel);
    normalized[channel] = value;
    sum += value * value;
  }
  if (norm_params.enabled[norm] != 0) {
    float inverse_rms = 1.0 / sqrt(sum / float(HEAD_DIM) + norm_params.epsilon[norm]);
    bool affine = is_query ? has_query_norm_scales : (is_key ? has_key_norm_scales : has_value_norm_scales);
    for (uint32_t channel = 0; channel < HEAD_DIM; ++channel) {
      float value = normalized[channel] * inverse_rms;
      if (affine) {
        float weight = is_query ? query_norm_scales[channel] :
                       (is_key ? key_norm_scales[channel] : value_norm_scales[channel]);
        weight += norm_params.scale_offset[norm];
        value = norm_params.full_layer[norm] != 0 ? value * weight :
                tensorRound<ElementT>(value) * tensorRound<ElementT>(weight);
      }
      normalized[channel] = tensorRound<ElementT>(value);
    }
  }
  for (uint32_t channel = 0; channel < HEAD_DIM; ++channel) {
    float value = normalized[channel];
    if (has_rope && (is_query || is_key) && channel < rope_dim) {
      uint32_t half_dim = rope_dim / 2;
      uint32_t paired = channel < half_dim ? channel + half_dim : channel - half_dim;
      float other = normalized[paired] * (channel < half_dim ? -1.0 : 1.0);
      value = tensorRound<ElementT>(
          value * tensorLoad<RopeT>(cosines, token_idx * rope_dim + channel) +
          other * tensorLoad<RopeT>(sines, token_idx * rope_dim + channel)
      );
    }
    values[channel] = value * attentionSign(channel);
  }
  attentionHadamard<HEAD_DIM>(values);
  float maximum = 0.0;
  for (uint32_t channel = 0; channel < HEAD_DIM; ++channel)
    maximum = max(maximum, abs(values[channel]));
  float scale = tensorRound<BFloat16>(maximum > 0.0 ? maximum / 127.0 : 1.0);
  uint32_t scale_index;
  if (is_query) {
    uint32_t gqa = num_q_heads / num_kv_heads;
    scale_index = (head_idx / gqa) * query_rows + (head_idx % gqa) * suffix_length + token_idx;
    tensorStore<BFloat16>(q_scales, scale_index, scale);
  } else {
    uint32_t kv_head = head_idx - num_q_heads - (is_key ? 0 : num_kv_heads);
    scale_index = (kv_token_offset + token_idx) * num_kv_heads + kv_head;
    if (is_key)
      tensorStore<BFloat16>(k_scales, scale_index, scale);
    else
      tensorStore<BFloat16>(v_scales, scale_index, scale);
  }
  for (uint32_t channel = 0; channel < HEAD_DIM; ++channel) {
    int32_t code = attentionQuantize(values[channel], scale);
    uint32_t index = scale_index * HEAD_DIM + channel;
    if (is_query)
      byteStore(q_codes, index, uint32_t(code));
    else if (is_key)
      byteStore(k_codes, index, uint32_t(code));
    else
      byteStore(v_codes, index, uint32_t(code + 128));
  }
}
