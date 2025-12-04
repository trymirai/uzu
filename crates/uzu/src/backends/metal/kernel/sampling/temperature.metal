#include <metal_stdlib>
#include "../definitions.metal"

#define BLOCK_SIZE 256
#define GRAIN_SIZE 16

SPECIALIZE(T, float, half, bfloat)

KERNEL(Temperature) (
    device const T* logits_buffer,
    device T* processed_logits_buffer,
    constant uint& batch_size,
    constant uint& vocab_size,
    constant float& temperature,
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
            processed_logits_buffer[global_idx] = T(float(logits_buffer[global_idx]) / temperature);
        }
    }
}
