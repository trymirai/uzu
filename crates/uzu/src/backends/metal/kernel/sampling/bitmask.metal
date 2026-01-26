#include <metal_stdlib>
#include "../definitions.metal"

#define BLOCK_SIZE 1024
#define GRAIN_SIZE 64
#define BITS_IN_U32 32

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(Bitmask) (
    device const T* logits,
    device const uint32_t* bitmask,
    constant uint& bitmask_offset,
    device T* processed_logits,
    constant uint& batch_size,
    constant uint& vocab_size,
    uint group_idx GROUPS(vocab_size.div_ceil(BLOCK_SIZE * GRAIN_SIZE)),
    uint batch_idx GROUPS(batch_size),
    uint thread_idx THREADS(BLOCK_SIZE)
) {
  bitmask += bitmask_offset / sizeof(uint32_t);
  uint bitmask_size = (vocab_size + (BITS_IN_U32 - 1)) / BITS_IN_U32;
  uint base_idx =
      batch_idx * vocab_size + group_idx * BLOCK_SIZE * GRAIN_SIZE + thread_idx;
  uint batch_end = batch_idx * vocab_size + vocab_size;

#pragma unroll(4)
  for (uint i = 0; i < GRAIN_SIZE; i++) {
    uint global_idx = base_idx + i * BLOCK_SIZE;
    if (global_idx < batch_end) {
      uint token_idx = global_idx - batch_idx * vocab_size;
      uint bitmask_idx = batch_idx * bitmask_size + (token_idx / BITS_IN_U32);
      bool mask = (bitmask[bitmask_idx] >> (token_idx % BITS_IN_U32)) & 0b1;
      processed_logits[global_idx] =
          select(T(-INFINITY), logits[global_idx], mask);
    }
  }
}
