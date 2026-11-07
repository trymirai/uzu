import definitions;
import storage;
import attention.common;
import generated.ring;
import generated.trie;

[[Variants("T", "float, BFloat16")]]
[[Variants("HEAD_DIM", "64, 128, 256, 512")]]
[[Kernel]] func AttentionSimple<T : ITensorElement, let HEAD_DIM : uint32_t>(
    StructuredBuffer<uint32_t> q_codes,
    StructuredBuffer<uint32_t> q_scales,
    StructuredBuffer<uint32_t> k_codes,
    StructuredBuffer<uint32_t> k_scales,
    StructuredBuffer<uint32_t> v_codes,
    StructuredBuffer<uint32_t> v_scales,
    [[Optional("is_trie")]] StructuredBuffer<TrieNode> trie,
    [[Optional("is_ring")]] RingParams ring,
    [[Optional("has_window")]] uint32_t window,
    [[Optional("has_sinks")]] StructuredBuffer<uint32_t> sinks,
    [[Optional("has_gate")]] StructuredBuffer<uint32_t> gate,
    [[Optional("has_gate")]] uint32_t gate_stride,
    RWStructuredBuffer<Atomic<uint32_t>> output,
    uint32_t prefix_length,
    uint32_t suffix_length,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    float scale,
    [[Specialize]] bool is_causal,
    [[Specialize]] bool is_trie,
    [[Specialize]] bool is_ring,
    [[Specialize]] bool has_window,
    [[Specialize]] bool has_sinks,
    [[Specialize]] bool has_gate,
    [[Groups("num_q_heads")]] uint32_t head,
    [[Groups("suffix_length")]] uint32_t query,
    [[Threads("1")]] uint32_t lane,
) {
  // One invocation computes a complete attention row in the CPU's row-major layout.
  uint32_t q_row = head * suffix_length + query;
  uint32_t kv_head = head / (num_q_heads / num_kv_heads);
  uint32_t suffix_position = is_ring ? ring.ring_length : prefix_length;
  uint32_t query_position = suffix_position + (is_trie ? trie[query].height : query);
  float query_scale = tensorLoad<BFloat16>(q_scales, q_row);
  // -FLT_MAX, not -inf: WGSL rejects shaders with a constant expression that evaluates to infinity.
  float maximum = has_sinks ? tensorLoad<T>(sinks, head) : asfloat(0xff7fffffu);
  float denominator = has_sinks ? 1.0 : 0.0;
  float values[HEAD_DIM];
  for (uint32_t channel = 0; channel < HEAD_DIM; ++channel)
    values[channel] = 0.0;

  for (uint32_t key = 0; key < prefix_length + suffix_length; ++key) {
    bool visible = true;
    uint32_t key_position;
    if (key >= prefix_length) {
      uint32_t suffix_key = key - prefix_length;
      key_position = suffix_position + (is_trie ? trie[suffix_key].height : suffix_key);
      if (is_causal)
        visible = is_trie ? query >= trie[suffix_key].trie_start && query <= trie[suffix_key].trie_end :
                           suffix_key <= query;
    } else if (is_ring) {
      key_position = (prefix_length + key - ring.ring_offset) % prefix_length;
      visible = key_position < ring.ring_length;
    } else {
      key_position = key;
    }
    if (has_window) {
      if (is_causal)
        visible = visible && key_position <= query_position && query_position - key_position < window;
      else
        visible = visible && (key_position <= query_position ? query_position - key_position :
                              key_position - query_position) <= window / 2;
    }
    if (!visible)
      continue;

    uint32_t kv_row = key * num_kv_heads + kv_head;
    int32_t dot = 0;
    for (uint32_t channel = 0; channel < HEAD_DIM; ++channel) {
      int32_t q = attentionSignedCode(byteLoad(q_codes, q_row * HEAD_DIM + channel));
      int32_t k = attentionSignedCode(byteLoad(k_codes, kv_row * HEAD_DIM + channel));
      dot += q * k;
    }
    float score = float(dot) * query_scale * tensorLoad<BFloat16>(k_scales, kv_row) * scale;
    float next_maximum = max(maximum, score);
    float correction = exp(maximum - next_maximum);
    float weight = exp(score - next_maximum);
    float value_scale = tensorLoad<BFloat16>(v_scales, kv_row);
    denominator = denominator * correction + weight;
    for (uint32_t channel = 0; channel < HEAD_DIM; ++channel) {
      float value = float(int32_t(byteLoad(v_codes, kv_row * HEAD_DIM + channel)) - 128) * value_scale;
      values[channel] = values[channel] * correction + weight * value;
    }
    maximum = next_maximum;
  }
  if (denominator > 0.0) {
    for (uint32_t channel = 0; channel < HEAD_DIM; ++channel)
      values[channel] /= denominator;
  }
  attentionHadamard<HEAD_DIM>(values);
  for (uint32_t channel = 0; channel < HEAD_DIM; ++channel) {
    float value = values[channel] * attentionSign(channel);
    if (has_gate) {
      float g = tensorLoad<T>(gate, query * gate_stride + head * HEAD_DIM + channel);
      value = tensorRound<T>(value) / (1.0 + exp(-g));
    }
    tensorStore<T>(output, (query * num_q_heads + head) * HEAD_DIM + channel, value);
  }
}
