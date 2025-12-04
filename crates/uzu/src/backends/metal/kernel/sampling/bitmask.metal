#include <metal_stdlib>
#include "../definitions.metal"

#define BLOCK_SIZE 256
#define GRAIN_SIZE 16
#define BITS_IN_U32 32

SPECIALIZE(T, float, half, bfloat)

KERNEL(Bitmask) (
    device const T* logits_buffer,
    device const uint32_t* bitmask_buffer,
    device T* processed_logits_buffer,
    constant uint& batch_size,
    constant uint& vocab_size,
    uint3 global_idx GLOBAL(vocab_size.div_ceil(BLOCK_SIZE * GRAIN_SIZE), batch_size),
    uint3 local_idx LOCAL(BLOCK_SIZE)
) {
    uint group_idx = global_idx.x;
    uint batch_idx = global_idx.y;
    uint thread_idx = local_idx.x;

    uint base_idx = batch_idx * vocab_size + group_idx * BLOCK_SIZE * GRAIN_SIZE + thread_idx;
    uint batch_end = batch_idx * vocab_size + vocab_size;

    #pragma unroll(4)
    for (uint i = 0; i < GRAIN_SIZE; i++) {
        uint global_idx = base_idx + i * BLOCK_SIZE;
        if (global_idx < batch_end) {
            bool mask = (bitmask_buffer[global_idx / BITS_IN_U32] >> (global_idx % BITS_IN_U32)) & 0b1;
            processed_logits_buffer[global_idx] = select(T(-INFINITY), logits_buffer[global_idx], mask);
        }
    }
}
