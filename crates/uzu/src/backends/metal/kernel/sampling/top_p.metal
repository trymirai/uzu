#include <metal_stdlib>
#include "../definitions.metal"

#define MAX_VOCAB_SIZE 262144
#define BLOCK_SIZE 1024
#define MAX_ITERS 64
#define MAX_SAMPLES_PER_BLOCK ((MAX_VOCAB_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE)
#define ATOL 1e-4

#define SEARCH_TYPE_INTERPOLATION

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
    const uint num_local_samples = (thread_idx < vocab_size) ? ((vocab_size - thread_idx + BLOCK_SIZE - 1) / BLOCK_SIZE) : 0;
    float local_probas[MAX_SAMPLES_PER_BLOCK];

    float local_norm = 0.0f;

    float local_max_logit = -INFINITY;
    #pragma unroll(4)
    for (uint i = thread_idx; i < vocab_size; i += BLOCK_SIZE) {
        local_max_logit = max(float(logits_data[batch_start + i]), local_max_logit);
    }
    const float max_logit = threadgroup_cooperative_reduce_max<BLOCK_SIZE>(
        local_max_logit,
        shared_float_reduce_buffer,
        thread_idx
    );

    #pragma unroll(4)
    for (uint i = thread_idx; i < vocab_size; i += BLOCK_SIZE) {
        const float logit_value = float(logits_data[batch_start + i]) - max_logit;
        const float local_proba = fast::exp(logit_value);
        local_probas[i / BLOCK_SIZE] = local_proba;
        local_norm += local_proba;
    }
    const float global_norm = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(
        local_norm,
        shared_float_reduce_buffer,
        thread_idx
    );

    #pragma unroll(4)
    for (uint i = 0; i < num_local_samples; ++i) {
        local_probas[i] /= global_norm;
    }

    float low = 0.0f;
    float high = 1.0f;
    float local_mass_above_high = 0.0f;

    float mass_above_high = 0.0f;
    float mass_above_low = 1.0f;

    uint local_block_start = 0;
    uint local_block_end = num_local_samples;

    for (uint iter = 0; iter < MAX_ITERS; ++iter) {
        const float slope = (mass_above_high - mass_above_low) / (high - low);

#if defined(SEARCH_TYPE_INTERPOLATION)
        const float threshold = (mass_above_high - top_p) / slope + low;
#elif defined(SEARCH_TYPE_BINARY_SEARCH)
        const float threshold = (high + low) / 2.0f;
#else
#error "Unsupported search type"
#endif

        float local_mass_above_threshold = local_mass_above_high;
        uint partition_left = local_block_start;
        uint partition_right = local_block_end;

        while (partition_left < partition_right) {
            const float proba = local_probas[partition_left];
            if (proba >= threshold) {
                local_mass_above_threshold += proba;

                --partition_right;
                const float tmp = local_probas[partition_right];
                local_probas[partition_right] = proba;
                local_probas[partition_left] = tmp;
            } else {
                ++partition_left;
            }
        }

        const uint local_search_size = partition_right - partition_left;
        const uint global_search_size = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(
            local_search_size,
            shared_uint_reduce_buffer,
            thread_idx
        );
        const float mass_above_threshold = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(
            local_mass_above_threshold,
            shared_float_reduce_buffer,
            thread_idx
        );

        if (mass_above_threshold >= top_p) {
            low = threshold;
            mass_above_low = mass_above_threshold;

            local_block_start = partition_left;
        } else {
            high = threshold;
            local_mass_above_high = local_mass_above_threshold;
            mass_above_high = mass_above_threshold;

            local_block_end = partition_right;
        }

        if (global_search_size <= 1 || high - low < ATOL) break;
    }

    #pragma unroll(4)
    for (uint i = thread_idx; i < vocab_size; i += BLOCK_SIZE) {
        const T logit_value = logits_data[batch_start + i];
        const float proba = fast::exp(float(logit_value) - max_logit) / global_norm;
        if (proba < low) {
            processed_logits[batch_start + i] = T(-INFINITY);
        } else {
            processed_logits[batch_start + i] = logit_value;
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
