#include <metal_stdlib>
#include "../definitions.metal"
#include "../rng.metal"

#define BLOCK_SIZE 1024
#define GRAIN_SIZE 64
#define WORDS_PER_OFFSET 4

SPECIALIZE(T, float, half, bfloat) KERNEL(Gumbel) (
    device const T* logits,
    device const uint64_t* batch_seeds,
    device T* processed_logits,
    constant uint& batch_size,
    constant uint& vocab_size,
    constant uint& seeds_offset,
    uint group_idx GROUPS(vocab_size.div_ceil(BLOCK_SIZE * GRAIN_SIZE)),
    uint batch_idx GROUPS(batch_size),
    uint thread_idx THREADS(BLOCK_SIZE)
) {
  uint batch_start = vocab_size * batch_idx;
  uint batch_end = batch_start + vocab_size;

  uint grain_offset_in_batch = group_idx * BLOCK_SIZE * GRAIN_SIZE + thread_idx;

  uint grain_offset = batch_start + grain_offset_in_batch;

  uint64_t rng_seed = batch_seeds[seeds_offset + batch_idx];
  uint64_t rng_offset = (group_idx * BLOCK_SIZE + thread_idx) *
                        (GRAIN_SIZE + WORDS_PER_OFFSET - 1) / WORDS_PER_OFFSET;
  PhiloxState rng;
  philox_init(&rng, rng_seed, rng_offset);

#pragma unroll(4)
  for (uint i = 0; i < GRAIN_SIZE; i++) {
    uint global_idx = grain_offset + i * BLOCK_SIZE;
    if (global_idx < batch_end) {
      processed_logits[global_idx] =
          logits[global_idx] + T(-fast::log(-fast::log(uniform_float(&rng))));
    }
  }
}
