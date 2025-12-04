#include <metal_stdlib>
#include "../definitions.metal"
#include "../rng.metal"

#define BLOCK_SIZE 1024
#define GRAIN_SIZE 64

#define WORDS_PER_OFFSET 4

SPECIALIZE(T, float, half, bfloat)

template <typename T>
void batched_gumbel(
    device const T* logits_data,
    device const uint64_t* batch_seeds,
    device T* processed_logits,
    constant uint& vocab_size,
    constant uint& seeds_offset_elems,
    uint group_idx,
    uint batch_idx,
    uint thread_idx
) {
    uint batch_start = vocab_size * batch_idx;
    uint batch_end = batch_start + vocab_size;

    uint grain_offset_in_batch = group_idx * BLOCK_SIZE * GRAIN_SIZE + thread_idx;

    uint grain_offset = batch_start + grain_offset_in_batch;

    uint64_t rng_seed = batch_seeds[seeds_offset_elems + batch_idx];
    uint64_t rng_offset = (group_idx * BLOCK_SIZE + thread_idx) * (GRAIN_SIZE + WORDS_PER_OFFSET - 1) / WORDS_PER_OFFSET;
    PhiloxState rng;
    philox_init(&rng, rng_seed, rng_offset);

    #pragma unroll(4)
    for (uint i = 0; i < GRAIN_SIZE; i++) {
        uint global_idx = grain_offset + i * BLOCK_SIZE;
        if (global_idx < batch_end) {
            processed_logits[global_idx] = logits_data[global_idx] + T(-fast::log(-fast::log(uniform_float(&rng))));
        }
    }
}

KERNEL(Gumbel) (
    device const T* logits_buffer,
    device const uint64_t* batch_seeds,
    device T* processed_logits_buffer,
    constant uint& batch_size,
    constant uint& vocab_size,
    constant uint& seeds_offset_elems,
    uint3 global_idx GLOBAL(vocab_size.div_ceil(BLOCK_SIZE * GRAIN_SIZE), batch_size),
    uint3 local_idx LOCAL(BLOCK_SIZE)
) {
    uint group_idx = global_idx.x;
    uint batch_idx = global_idx.y;
    uint thread_idx = local_idx.x;
    batched_gumbel(
        logits_buffer,
        batch_seeds,
        processed_logits_buffer,
        vocab_size,
        seeds_offset_elems,
        group_idx,
        batch_idx,
        thread_idx
    );
}
