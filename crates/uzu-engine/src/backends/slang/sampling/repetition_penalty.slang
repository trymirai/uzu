import definitions;
import storage;

[[Variants("T", "float, half, BFloat16")]]
[[Public]] [[Kernel]] func RepetitionPenalty<T : ITensorElement>(
    StructuredBuffer<uint> original_logits,
    RWStructuredBuffer<Atomic<uint>> logits_copy,
    StructuredBuffer<uint> context_ring,
    StructuredBuffer<uint> token_ids,
    float repetition_penalty,
    uint suffix_repetition_length,
    uint vocab_size,
    uint sampling_start,
    uint sampling_length,
    [[Axis("sampling_length * suffix_repetition_length", "256")]] uint position,
) {
  uint3 gid = uint3(position % suffix_repetition_length, position / suffix_repetition_length, 0);

  let window_index = gid.x;
  let sample = gid.y;
  if (window_index >= suffix_repetition_length || sample >= sampling_length) {
    return;
  }
  let ring_offset = context_ring[0u];
  let ring_length = context_ring[1u];
  let total_length = ring_length + sampling_start + sample + 1u;
  let window_length = min(total_length, suffix_repetition_length);
  if (window_index >= window_length) {
    return;
  }
  let source = total_length - window_length + window_index;
  var token = 0u;
  if (source < ring_length) {
    token = context_ring[2u + (ring_offset + source) % suffix_repetition_length];
  } else {
    token = token_ids[source - ring_length];
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
