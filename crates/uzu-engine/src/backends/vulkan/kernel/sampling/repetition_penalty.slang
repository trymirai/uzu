import definitions;

static const uint REPETITION_PENALTY_THREADS = 256;
static const uint REPETITION_PENALTY_EMPTY = 0xffffffffu;
groupshared uint repetition_penalty_heads[256];
groupshared uint repetition_penalty_next[256];
groupshared uint repetition_penalty_tokens[256];

// One workgroup per sample; duplicates read original_logits, never a penalized output. Each token's hash owner writes it
// throughout a row; a bounded shared list deduplicates each chunk without racing device stores. Unchecked, as on the
// CPU: output does not alias inputs, tokens are below vocab_size, the ring is valid and its offsets do not overflow.
[[Kernel]] [[Public]]
func RepetitionPenalty<T: IModelFloat>(
  Ptr<T, Access.Read> original_logits,
  Ptr<T, Access.ReadWrite> logits_copy,
  Ptr<uint, Access.Read> context_ring,
  Ptr<uint, Access.Read> token_ids,
  float repetition_penalty,
  uint suffix_repetition_length,
  uint vocab_size,
  uint sampling_start,
  uint sampling_length,
  [[Groups("if suffix_repetition_length == 0 { 0 } else { sampling_length }")]] uint group,
  [[Threads("256")]] uint tid,
) {
  static_assert(REPETITION_PENALTY_THREADS == 256, "256 hash owners and chunk slots");
  let capacity = suffix_repetition_length;
  let ring_offset = context_ring[0];
  let ring_length = context_ring[1];
  let total_length = ring_length + sampling_start + group + 1;
  let window_length = min(total_length, capacity);
  let skipped_length = total_length - window_length;
  for (uint base = 0; base < window_length; base += REPETITION_PENALTY_THREADS) {
    repetition_penalty_heads[tid] = REPETITION_PENALTY_EMPTY;
    GroupMemoryBarrierWithGroupSync();
    if (tid < window_length - base) {
      let source_index = skipped_length + base + tid;
      uint token;
      if (source_index < ring_length) {
        token = context_ring[2 + (ring_offset + source_index) % capacity];
      } else {
        token = token_ids[source_index - ring_length];
      }
      repetition_penalty_tokens[tid] = token;
      uint previous;
      InterlockedExchange(repetition_penalty_heads[(token * 0x9e3779b9u) >> 24], tid, previous);
      repetition_penalty_next[tid] = previous;
    }
    // Every producer inserted once: predecessor links form a finite list. Publish tokens and links before traversal.
    GroupMemoryBarrierWithGroupSync();
    var node = repetition_penalty_heads[tid];
    while (node != REPETITION_PENALTY_EMPTY) {
      let token = repetition_penalty_tokens[node];
      var earlier = repetition_penalty_heads[tid];
      while (earlier != node && repetition_penalty_tokens[earlier] != token) {
        earlier = repetition_penalty_next[earlier];
      }
      if (earlier == node) {
        let index = group * vocab_size + token;
        let logit = original_logits[index].toFloat();
        let penalized = logit > 0.0 ? logit / repetition_penalty : logit * repetition_penalty;
        logits_copy[index] = T.fromFloat(penalized);
      }
      node = repetition_penalty_next[node];
    }
    GroupMemoryBarrierWithGroupSync();
  }
}
