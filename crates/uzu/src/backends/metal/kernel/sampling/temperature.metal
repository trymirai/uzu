#include <metal_stdlib>
#include "../definitions.metal"

#define BLOCK_SIZE 1024
#define GRAIN_SIZE 64

SPECIALIZE(T, float, half, bfloat) KERNEL(Temperature) (
    device const T* logits,
    device T* processed_logits,
    constant uint& batch_size,
    constant uint& vocab_size,
    constant float& temperature,
    uint group_idx GROUPS(vocab_size.div_ceil(BLOCK_SIZE * GRAIN_SIZE)),
    uint batch_idx GROUPS(batch_size),
    uint thread_idx THREADS(BLOCK_SIZE)
) {
  uint base_idx =
      batch_idx * vocab_size + group_idx * BLOCK_SIZE * GRAIN_SIZE + thread_idx;
  uint batch_end = batch_idx * vocab_size + vocab_size;

#pragma unroll(4)
  for (uint i = 0; i < GRAIN_SIZE; i++) {
    uint global_idx = base_idx + i * BLOCK_SIZE;
    if (global_idx < batch_end) {
      processed_logits[global_idx] = T(float(logits[global_idx]) / temperature);
    }
  }
}
