#include <metal_stdlib>
#include "../definitions.metal"

#define BLOCK_SIZE 1024
#define BLOCK_SIZE_IN_SIMDS (BLOCK_SIZE / 32)

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MinP) (
    device const T* logits,
    device T* processed_logits,
    threadgroup float shared_reduce_buffer[BLOCK_SIZE_IN_SIMDS],
    constant uint& batch_size,
    constant uint& vocab_size,
    constant float& min_p,
    const Simd simd,
    uint batch_idx GROUPS(batch_size),
    uint thread_idx THREADS(BLOCK_SIZE)
) {
  uint batch_start = batch_idx * vocab_size;

  // Find maximum logit
  float local_max = -INFINITY;
#pragma unroll(4)
  for (uint i = thread_idx; i < vocab_size; i += BLOCK_SIZE) {
    float logit_value = float(logits[batch_start + i]);
    local_max = fmax(local_max, logit_value);
  }
  float max_logit = threadgroup_cooperative_reduce_max<BLOCK_SIZE>(
      local_max,
      shared_reduce_buffer,
      thread_idx,
      simd
  );

  // Then the threshold is just max_logit + log(min_p), mask everything strictly
  // below it
  T t_threshold = T(max_logit + log(min_p));
#pragma unroll(4)
  for (uint i = thread_idx; i < vocab_size; i += BLOCK_SIZE) {
    T logit_value = logits[batch_start + i];
    processed_logits[batch_start + i] =
        select(T(-INFINITY), logit_value, logit_value >= t_threshold);
  }
}
