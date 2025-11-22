#include <metal_stdlib>
#include "../definitions.metal"

#define BLOCK_SIZE 1024
#define MAX_ITERS 8
#define BRANCHING_FACTOR 8
#define MAX_SAMPLES_PER_BLOCK (256000 / BLOCK_SIZE)

template <typename T>
void batched_topp(
    device const T* logits_data,
    device T* processed_logits,
    threadgroup float shared_float_reduce_buffer[BLOCK_SIZE],
    threadgroup uint shared_uint_reduce_buffer[BLOCK_SIZE],
    constant uint& vocab_size,
    constant float& top_p,
    uint batch_idx,
    uint thread_idx
) {
    const uint batch_start = batch_idx * vocab_size;
    const uint num_samples_per_block = (vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float local_probas[MAX_SAMPLES_PER_BLOCK];

    float local_norm = 0.0f;
    for (uint i = thread_idx; i < vocab_size; i += BLOCK_SIZE) {
        const float logit_value = float(logits_data[batch_start + i]);
        const float local_proba = exp(logit_value);
        local_probas[i / BLOCK_SIZE] = local_proba;
        local_norm += local_proba;
    }
    const float global_norm = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(
        local_norm,
        shared_float_reduce_buffer,
        thread_idx
    );
    for (uint i = 0; i < num_samples_per_block; ++i) {
        local_probas[i] /= global_norm;
    }

    float low = 0.0f;
    float high = top_p;

    for (uint iter = 0; iter < MAX_ITERS; ++iter) {
        const float step_size = (high - low) / BRANCHING_FACTOR;
        uint search_size;
        uint num_tokens_above_prev_threshold = vocab_size;
        for (uint branch = 0; branch < BRANCHING_FACTOR - 1; ++branch) {
            const float threshold = low + step_size * (branch + 1);

            float local_sum_above_threshold = 0.0f;
            uint local_num_tokens_above_threshold = 0;
            for (uint i = thread_idx; i < vocab_size; i += BLOCK_SIZE) {
                const float proba = local_probas[i / BLOCK_SIZE];
                if (proba >= threshold) {
                    local_sum_above_threshold += proba;
                    ++local_num_tokens_above_threshold;
                }
            }
            const float sum_above_threshold = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(
                local_sum_above_threshold,
                shared_float_reduce_buffer,
                thread_idx
            );
            const uint num_tokens_above_threshold = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(
                local_num_tokens_above_threshold,
                shared_uint_reduce_buffer,
                thread_idx
            );
            if (sum_above_threshold >= top_p) {
                low = threshold;
            } else {
                high = threshold;
                search_size = num_tokens_above_prev_threshold - num_tokens_above_threshold;
                break;
            }
            search_size = num_tokens_above_prev_threshold - num_tokens_above_threshold;
            num_tokens_above_prev_threshold = num_tokens_above_threshold;
        }

        if (search_size == 0) break;
    }

    // We know the threshold, just mask everything below it
    for (uint i = thread_idx; i < vocab_size; i += BLOCK_SIZE) {
        const float proba = local_probas[i / BLOCK_SIZE];
        if (proba < low) {
            processed_logits[batch_start + i] = T(-INFINITY);
        }
    }
}

#define outerArguments(T) \
(device const T* logits_data [[ buffer(0) ]], \
device T* processed_logits [[ buffer(1) ]], \
constant uint& vocab_size [[ buffer(2) ]], \
constant float& top_p [[ buffer(3) ]], \
uint3 threadgroup_idx [[ threadgroup_position_in_grid ]], \
uint3 thread_idx [[ thread_position_in_threadgroup ]])

#define innerArguments (logits_data, processed_logits, shared_float_reduce_buffer, shared_uint_reduce_buffer, vocab_size, top_p, threadgroup_idx.x, thread_idx.x)

#define generateToppKernel(functionName, scalarType, outerArgs, innerArgs) \
kernel void functionName##_##scalarType outerArgs {                        \
    threadgroup float shared_float_reduce_buffer[BLOCK_SIZE];              \
    threadgroup uint shared_uint_reduce_buffer[BLOCK_SIZE];                \
    functionName innerArgs;                                                \
}

#define generateToppKernels(functionName)                                        \
generateToppKernel(functionName, float, outerArguments(float), innerArguments);  \
generateToppKernel(functionName, bfloat, outerArguments(bfloat), innerArguments); \
generateToppKernel(functionName, half, outerArguments(half), innerArguments);

generateToppKernels(batched_topp)

#undef outerArguments
#undef innerArguments
#undef generateToppKernel
#undef generateToppKernels
