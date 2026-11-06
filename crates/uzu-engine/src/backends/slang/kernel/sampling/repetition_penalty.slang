import definitions;
import storage;
import generated.trie;

[[Variants("T", "float, half, BFloat16")]]
[[Public]] [[Kernel]] func RepetitionPenalty<T : ITensorElement>(
    StructuredBuffer<uint32_t> original_logits,
    RWStructuredBuffer<Atomic<uint32_t>> logits_copy,
    StructuredBuffer<uint32_t> context_ring,
    StructuredBuffer<uint32_t> token_ids,
    [[Optional("is_trie")]] StructuredBuffer<TrieNode> trie,
    float repetition_penalty,
    uint32_t suffix_repetition_length,
    uint32_t vocab_size,
    uint32_t sampling_start,
    uint32_t sampling_length,
    [[Specialize]] bool is_trie,
    [[Axis("sampling_length * suffix_repetition_length", "256")]] uint32_t position,
) {
  uint3 gid = uint3(position % suffix_repetition_length, position / suffix_repetition_length, 0);

  let window_index = gid.x;
  let sample = gid.y;
  if (window_index >= suffix_repetition_length || sample >= sampling_length) {
    return;
  }
  let ring_offset = context_ring[0u];
  let ring_length = context_ring[1u];
  let batch_index = sampling_start + sample;
  let batch_prefix_length = (is_trie ? trie[batch_index].height : batch_index) + 1u;
  let total_length = ring_length + batch_prefix_length;
  let window_length = min(total_length, suffix_repetition_length);
  if (window_index >= window_length) {
    return;
  }
  let source = total_length - window_length + window_index;
  var token = 0u;
  if (source < ring_length) {
    token = context_ring[2u + (ring_offset + source) % suffix_repetition_length];
  } else {
    var token_index = source - ring_length;
    if (is_trie) {
      let height = token_index;
      token_index = batch_index;
      while (trie[token_index].height != height) {
        --token_index;
      }
    }
    token = token_ids[token_index];
  }
  if (token >= vocab_size) {
    return;
  }
  let offset = sample * vocab_size + token;
  let logit = tensorLoad<T>(original_logits, offset);
  var penalized = logit * repetition_penalty;
  if (logit > 0.0) {
    penalized = logit / repetition_penalty;
  }
  tensorStore<T>(logits_copy, offset, penalized);
}
