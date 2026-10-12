import definitions;
import common.activation;
import common.ring_index;
import common.storage_address;

static const uint REPETITION_PENALTY_THREADS = 256;
static const uint REPETITION_PENALTY_EMPTY = 0xffffffffu;
groupshared uint repetition_penalty_heads[256];
groupshared uint repetition_penalty_next[256];
groupshared uint repetition_penalty_tokens[256];

// Output does not alias inputs; each token is below vocab_size, the ring is valid, and start + length does not
// overflow (the CPU contract). Duplicates read original_logits, never a previously penalized output. Each token's
// hash owner writes it throughout a row; a bounded shared list deduplicates each chunk without racing device stores.
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
  [[Groups("if sampling_length == 0 || suffix_repetition_length == 0 { 0 } else { sampling_length.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0]) }")]] uint group,
  [[Threads("256")]] uint tid,
) {
  static_assert(REPETITION_PENALTY_THREADS == 256, "256 hash owners and chunk slots");
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let capacity = suffix_repetition_length;
  let ring_offset = context_ring[0];
  let ring_length = context_ring[1];
  let ring_tokens = element_pointer(context_ring, uint2(2, 0));
  var sample = group;
  while (sample < sampling_length) {
    // Split the suffix into an old-ring tail and a batch-prefix tail without adding ring_length to the prefix.
    let last = sampling_start + sample;
    let tail = min(last, capacity - 1) + 1;
    let first_token = last - (tail - 1);
    let ring_take = min(ring_length, capacity - tail);
    let ring_first = ring_length - ring_take;
    let count = ring_take + tail;
    var base = 0u;
    while (base < count) {
      repetition_penalty_heads[tid] = REPETITION_PENALTY_EMPTY;
      GroupMemoryBarrierWithGroupSync();
      if (tid < count - base) {
        let window = base + tid;
        uint token;
        if (window < ring_take) {
          let slot = ring_add(ring_offset, ring_first + window, capacity);
          token = element_pointer(ring_tokens, uint2(slot, 0))[0];
        } else {
          token = element_pointer(token_ids, uint2(first_token + (window - ring_take), 0))[0];
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
          let index = element_index(sample, vocab_size, token);
          let logit = element_pointer(original_logits, index)[0].toFloat();
          let positive = asuint(logit) - 1u < 0x7f800000u;
          let penalized = positive ? exact_quotient(logit, repetition_penalty) : exact_product(logit, repetition_penalty);
          element_pointer(logits_copy, index)[0] = T.fromFloat(penalized);
        }
        node = repetition_penalty_next[node];
      }
      GroupMemoryBarrierWithGroupSync();
      if (count - base <= REPETITION_PENALTY_THREADS) {
        break;
      }
      base += REPETITION_PENALTY_THREADS;
    }
    if (sampling_length - sample <= groups.x) {
      break;
    }
    sample += groups.x;
  }
}
