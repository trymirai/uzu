#include <metal_stdlib>
#include "../definitions.metal"

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(RepetitionPenalty)(
    device T* logits,
    device const uint32_t* previous_tokens,
    device const uint32_t* previous_counts,
    constant uint& batch_size,
    constant uint& vocab_size,
    constant uint& max_previous_tokens,
    constant float& repetition_penalty,
    uint batch_idx GROUPS(batch_size),
    uint thread_idx THREADS(64)
) {
  uint count = min(previous_counts[batch_idx], max_previous_tokens);
  if (thread_idx >= count) {
    return;
  }

  uint token = previous_tokens[batch_idx * max_previous_tokens + thread_idx];
  if (token >= vocab_size) {
    return;
  }

  uint index = batch_idx * vocab_size + token;
  float score = float(logits[index]);
  float adjusted = score < 0.0f ? score * repetition_penalty
                                : score / repetition_penalty;
  logits[index] = T(adjusted);
}
