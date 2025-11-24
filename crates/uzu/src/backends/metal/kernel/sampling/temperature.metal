#include <metal_stdlib>
#include "../definitions.metal"

#define GRAIN_SIZE 64
#define BLOCK_SIZE 1024

template <typename T>
void batched_temperature(
    device const T* logits_data,
    device T* processed_logits,
    constant uint& vocab_size,
    constant float& temperature,
    uint group_idx,
    uint batch_idx,
    uint thread_idx
) {
    uint base_idx = batch_idx * vocab_size + group_idx * BLOCK_SIZE * GRAIN_SIZE + thread_idx;
    uint batch_end = batch_idx * vocab_size + vocab_size;

    #pragma unroll(4)
    for (uint i = 0; i < GRAIN_SIZE; i++) {
        uint global_idx = base_idx + i * BLOCK_SIZE;
        if (global_idx < batch_end) {
            processed_logits[global_idx] = T(float(logits_data[global_idx]) / temperature);
        }
    }
}

#define outerArguments(T) \
(device const T* logits_data [[ buffer(0) ]], \
device T* processed_logits [[ buffer(1) ]], \
constant uint& vocab_size [[ buffer(2) ]], \
constant float& temperature [[ buffer(3) ]], \
uint3 threadgroup_idx [[ threadgroup_position_in_grid ]], \
uint3 thread_idx [[ thread_position_in_threadgroup ]])

#define innerArguments (logits_data, processed_logits, vocab_size, temperature, threadgroup_idx.x, threadgroup_idx.y, thread_idx.x)

generateKernels(batched_temperature)

#undef outerArguments
#undef innerArguments
