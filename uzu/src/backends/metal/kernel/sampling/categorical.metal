#include <metal_stdlib>
#include <metal_atomic>
#include "../definitions.metal"
#include "../rng.metal"
using namespace metal;

#define BLOCK_SIZE 1024
#define GRAIN_SIZE 4

struct CategoricalTempStorage {
    _atomic<uint> sampled_id;
    float         random_threshold;
};

template<typename T>
void categorical_sampling_kernel_impl(
    const device T* logits,
    device uint*    out_ids,
    constant uint&  batch,
    constant uint&  vocab,
    constant float& temperature,
    constant uint64_t& seed,
    threadgroup CategoricalTempStorage& ts,
    threadgroup float* shared_max,
    threadgroup float* shared_sum,
    uint batch_idx,
    uint lid)
{
    if (batch_idx >= batch) return;

    const uint stride_elems   = BLOCK_SIZE * GRAIN_SIZE;
    const uint num_vec_chunks = (vocab + stride_elems - 1) / stride_elems;
    const uint row_off        = batch_idx * vocab;

    PhiloxState rng;
    uint64_t philox_offset = batch_idx;
    philox_init(&rng, seed, philox_offset);

    /* Initialize shared memory */
    if (lid == 0) {
        atomic_store_explicit(&ts.sampled_id, vocab, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* ── Step 1: Find max logit for numerical stability ── */
    float local_max = -INFINITY;
    for (uint i = lid; i < vocab; i += BLOCK_SIZE) {
        local_max = max(local_max, float(logits[row_off + i]));
    }
    float max_logit = threadgroup_cooperative_reduce_max<BLOCK_SIZE>(local_max, shared_max, lid);

    /* ── Step 2: Compute sum of exp((logit - max_logit) / temperature) for normalization ── */
    float safe_temperature = max(temperature, 1e-8f);
    
    float local_sum = 0.0f;
    for (uint i = lid; i < vocab; i += BLOCK_SIZE) {
        local_sum += exp((float(logits[row_off + i]) - max_logit) / safe_temperature);
    }
    float sum_exp = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(local_sum, shared_sum, lid);

    /* ── Step 3: Generate random threshold ── */
    if (lid == 0) {
        ts.random_threshold = uniform_float(&rng);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float random_threshold = ts.random_threshold;

    /* ── Step 4: Find the token where cumulative probability crosses threshold ── */
    float running_prob_sum = 0.0f;
    
    for (uint vec_i = 0; vec_i < num_vec_chunks; ++vec_i) {
        /* Load probabilities for this chunk */
        float prob_vec[GRAIN_SIZE];
        bool valid[GRAIN_SIZE];
        
        for (uint j = 0; j < GRAIN_SIZE; ++j) {
            uint global_token_id = (vec_i * BLOCK_SIZE + lid) * GRAIN_SIZE + j;
            if (global_token_id < vocab) {
                prob_vec[j] = exp((float(logits[row_off + global_token_id]) - max_logit) / safe_temperature) / sum_exp;
                valid[j] = true;
            } else {
                prob_vec[j] = 0.0f;
                valid[j] = false;
            }
        }

        /* Compute block-wide sum for this chunk */
        float local_chunk_sum = prob_vec[0] + prob_vec[1] + prob_vec[2] + prob_vec[3];
        float block_chunk_sum = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(local_chunk_sum, shared_sum, lid);

        /* Check if our target is in this chunk */
        if (running_prob_sum + block_chunk_sum > random_threshold) {
            /* Compute prefix sum within thread's elements */
            float cumulative_probs[GRAIN_SIZE];
            cumulative_probs[0] = prob_vec[0];
            for (uint j = 1; j < GRAIN_SIZE; ++j) {
                cumulative_probs[j] = cumulative_probs[j-1] + prob_vec[j];
            }
            
            /* Get prefix sum from previous threads */
            float thread_prefix = threadgroup_raking_prefix_exclusive_sum<BLOCK_SIZE>(local_chunk_sum, shared_sum, lid);
            
            /* Add global running sum and thread prefix */
            for (uint j = 0; j < GRAIN_SIZE; ++j) {
                cumulative_probs[j] += running_prob_sum + thread_prefix;
            }

            /* Find first element that crosses the threshold */
            for (uint j = 0; j < GRAIN_SIZE; ++j) {
                if (valid[j] && cumulative_probs[j] > random_threshold) {
                    uint global_token_id = (vec_i * BLOCK_SIZE + lid) * GRAIN_SIZE + j;
                    atomic_fetch_min_explicit(&ts.sampled_id, global_token_id, memory_order_relaxed);
                    break;
                }
            }
            break;
        }

        running_prob_sum += block_chunk_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Output the result */
    if (lid == 0) {
        uint sampled = atomic_load_explicit(&ts.sampled_id, memory_order_relaxed);
        // Fallback to last valid token if something went wrong
        if (sampled >= vocab) {
            sampled = vocab - 1;
        }
        out_ids[batch_idx] = sampled;
    }
}

// Generate categorical single-pass kernels using macro approach
#define outerArguments(T) \
(device const T* logits_data [[ buffer(0) ]], \
device uint* sampled_tokens [[ buffer(1) ]], \
constant uint& batch_size [[ buffer(2) ]], \
constant uint& vocab_size [[ buffer(3) ]], \
constant float& temperature [[ buffer(4) ]], \
constant uint64_t& seed [[ buffer(5) ]], \
uint batch_idx [[ threadgroup_position_in_grid ]], \
uint local_id [[ thread_position_in_threadgroup ]])

#define innerArguments \
(logits_data, sampled_tokens, batch_size, vocab_size, temperature, seed, temp_storage, shared_max, shared_sum, batch_idx, local_id)

#define generateCategoricalKernel(functionName, T, outerArgs, innerArgs) \
kernel void functionName##_##T outerArgs { \
    threadgroup CategoricalTempStorage temp_storage; \
    threadgroup float shared_max[BLOCK_SIZE]; \
    threadgroup float shared_sum[BLOCK_SIZE]; \
    categorical_sampling_kernel_impl<T> innerArgs; \
}

generateCategoricalKernel(batched_categorical_main, float, outerArguments(float), innerArguments)
generateCategoricalKernel(batched_categorical_main, half, outerArguments(half), innerArguments)
generateCategoricalKernel(batched_categorical_main, bfloat, outerArguments(bfloat), innerArguments)

#undef outerArguments
#undef innerArguments
#undef generateCategoricalKernel

// MARK: - 2-Pass Categorical Sampling

struct CategoricalChunkResult {
    float max_logit;
    float sum_exp;
    float cumulative_prob;
};

template<typename T>
void categorical_sampling_main_2pass_impl(
    const device T* logits,
    device CategoricalChunkResult* chunk_results,
    constant uint& batch,
    constant uint& vocab,
    constant float& temperature,
    constant uint64_t& seed,
    threadgroup float* shared_max,
    threadgroup float* shared_sum,
    uint batch_idx,
    uint chunk_idx,
    uint lid)
{
    if (batch_idx >= batch) return;

    const uint stride_elems = BLOCK_SIZE * GRAIN_SIZE;
    const uint chunk_start = chunk_idx * stride_elems;
    const uint chunk_end = min(chunk_start + stride_elems, vocab);
    const uint row_off = batch_idx * vocab;

    // Step 1: Find max logit in this chunk
    float local_max = -INFINITY;
    for (uint i = chunk_start + lid; i < chunk_end; i += BLOCK_SIZE) {
        local_max = max(local_max, float(logits[row_off + i]));
    }
    float chunk_max = threadgroup_raking_reduce_max<BLOCK_SIZE>(local_max, shared_max, lid);

    // Step 2: Compute sum of exp((logit - max_logit) / temperature) for this chunk
    float safe_temperature = max(temperature, 1e-8f);
    
    float local_sum = 0.0f;
    for (uint i = chunk_start + lid; i < chunk_end; i += BLOCK_SIZE) {
        local_sum += exp((float(logits[row_off + i]) - chunk_max) / safe_temperature);
    }
    float chunk_sum = threadgroup_raking_reduce_sum<BLOCK_SIZE>(local_sum, shared_sum, lid);

    // Store results for this chunk
    if (lid == 0) {
        uint vocab_groups_per_batch = (vocab + stride_elems - 1) / stride_elems;
        uint result_idx = batch_idx * vocab_groups_per_batch + chunk_idx;
        chunk_results[result_idx] = {chunk_max, chunk_sum, 0.0f}; // cumulative_prob set in final pass
    }
}

template<typename T>
void categorical_sampling_final_2pass_impl(
    const device T* logits,
    const device CategoricalChunkResult* chunk_results,
    device uint* out_ids,
    constant uint& batch,
    constant uint& vocab,
    constant uint& num_chunks,
    constant float& temperature,
    constant uint64_t& seed,
    threadgroup float* shared_storage,
    threadgroup float* shared_sum,
    uint batch_idx,
    uint lid)
{
    if (batch_idx >= batch) return;

    const uint chunk_offset = batch_idx * num_chunks;
    const uint row_off = batch_idx * vocab;
    
    // Find global max across all chunks
    float global_max = -INFINITY;
    for (uint i = lid; i < num_chunks; i += BLOCK_SIZE) {
        global_max = max(global_max, chunk_results[chunk_offset + i].max_logit);
    }
    global_max = threadgroup_raking_reduce_max<BLOCK_SIZE>(global_max, shared_storage, lid);

    // Compute global sum with proper normalization
    float global_sum = 0.0f;
    for (uint i = lid; i < num_chunks; i += BLOCK_SIZE) {
        const device CategoricalChunkResult& chunk = chunk_results[chunk_offset + i];
        float renorm_factor = exp(chunk.max_logit - global_max);
        global_sum += chunk.sum_exp * renorm_factor;
    }
    global_sum = threadgroup_raking_reduce_sum<BLOCK_SIZE>(global_sum, shared_storage, lid);

    // Generate random threshold and find target chunk
    threadgroup uint* shared_target_chunk = (threadgroup uint*)shared_storage;
    threadgroup float* shared_target_within = (threadgroup float*)(shared_storage + 1);
    
    if (lid == 0) {
        PhiloxState rng;
        uint64_t philox_offset = batch_idx;
        philox_init(&rng, seed, philox_offset);
        float random_threshold = uniform_float(&rng) * global_sum;

        // Find which chunk contains the target probability
        float cumulative_prob = 0.0f;
        uint target_chunk = num_chunks;  // fallback
        float target_within_chunk = 0.0f;
        
        for (uint chunk_i = 0; chunk_i < num_chunks; chunk_i++) {
            const device CategoricalChunkResult& chunk = chunk_results[chunk_offset + chunk_i];
            float chunk_prob = chunk.sum_exp * exp(chunk.max_logit - global_max);
            
            if (cumulative_prob + chunk_prob > random_threshold) {
                target_chunk = chunk_i;
                // Calculate relative position within the chunk (0.0 to 1.0)
                target_within_chunk = (random_threshold - cumulative_prob) / chunk_prob;
                break;
            }
            cumulative_prob += chunk_prob;
        }
        
        // Store in shared memory for all threads
        *shared_target_chunk = target_chunk;
        *shared_target_within = target_within_chunk;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint target_chunk = *shared_target_chunk;
    float target_within_chunk = *shared_target_within;
    
    // If no valid chunk found, fallback
    if (target_chunk >= num_chunks) {
        if (lid == 0) {
            out_ids[batch_idx] = vocab - 1;
        }
        return;
    }

    // only thread 0 does the scan
    if (lid == 0) {
        const uint stride_elems = BLOCK_SIZE * GRAIN_SIZE;
        const uint chunk_start = target_chunk * stride_elems;
        const uint chunk_end = min(chunk_start + stride_elems, vocab);
        const device CategoricalChunkResult& target_chunk_result = chunk_results[chunk_offset + target_chunk];
        
        float safe_temperature = max(temperature, 1e-8f);
        float chunk_threshold = target_within_chunk * target_chunk_result.sum_exp;
        
        // Sample within the chunk using cumulative distribution
        float running_sum = 0.0f;
        uint selected_token = chunk_end - 1; // fallback to last token in chunk
        
        // Only thread 0 scans the target chunk - eliminates 1023 redundant operations
        for (uint i = chunk_start; i < chunk_end; i++) {
            float prob = exp((float(logits[row_off + i]) - target_chunk_result.max_logit) / safe_temperature);
            running_sum += prob;
            
            if (running_sum > chunk_threshold) {
                selected_token = i;
                break;
            }
        }
        
        out_ids[batch_idx] = selected_token;
    }
}

// Generate categorical 2-pass main kernels using macro approach
#define outerArguments(T) \
(device const T* logits_data [[ buffer(0) ]], \
device CategoricalChunkResult* chunk_results [[ buffer(1) ]], \
constant uint& batch_size [[ buffer(2) ]], \
constant uint& vocab_size [[ buffer(3) ]], \
constant float& temperature [[ buffer(4) ]], \
constant uint64_t& seed [[ buffer(5) ]], \
uint2 group_position [[ threadgroup_position_in_grid ]], \
uint2 local_position [[ thread_position_in_threadgroup ]])

#define innerArguments \
(logits_data, chunk_results, batch_size, vocab_size, temperature, seed, shared_max, shared_sum, group_position.x, group_position.y, local_position.x)

#define generateCategorical2PassMainKernel(functionName, T, outerArgs, innerArgs) \
kernel void functionName##_##T outerArgs { \
    threadgroup float shared_max[BLOCK_SIZE]; \
    threadgroup float shared_sum[BLOCK_SIZE]; \
    categorical_sampling_main_2pass_impl<T> innerArgs; \
}

generateCategorical2PassMainKernel(batched_categorical_main_2pass, float, outerArguments(float), innerArguments)
generateCategorical2PassMainKernel(batched_categorical_main_2pass, half, outerArguments(half), innerArguments)
generateCategorical2PassMainKernel(batched_categorical_main_2pass, bfloat, outerArguments(bfloat), innerArguments)

#undef outerArguments
#undef innerArguments
#undef generateCategorical2PassMainKernel

// Generate categorical 2-pass final kernels using macro approach
#define outerArguments(T) \
(device const T* logits_data [[ buffer(0) ]], \
device const CategoricalChunkResult* chunk_results [[ buffer(1) ]], \
device uint* sampled_tokens [[ buffer(2) ]], \
constant uint& batch_size [[ buffer(3) ]], \
constant uint& vocab_size [[ buffer(4) ]], \
constant uint& num_chunks [[ buffer(5) ]], \
constant float& temperature [[ buffer(6) ]], \
constant uint64_t& seed [[ buffer(7) ]], \
uint batch_idx [[ threadgroup_position_in_grid ]], \
uint local_id [[ thread_position_in_threadgroup ]])

#define innerArguments \
(logits_data, chunk_results, sampled_tokens, batch_size, vocab_size, num_chunks, temperature, seed, shared_storage, shared_sum, batch_idx, local_id)

#define generateCategorical2PassFinalKernel(functionName, T, outerArgs, innerArgs) \
kernel void functionName##_##T outerArgs { \
    threadgroup float shared_storage[BLOCK_SIZE]; \
    threadgroup float shared_sum[BLOCK_SIZE]; \
    categorical_sampling_final_2pass_impl<T> innerArgs; \
}

generateCategorical2PassFinalKernel(batched_categorical_final_2pass, float, outerArguments(float), innerArguments)
generateCategorical2PassFinalKernel(batched_categorical_final_2pass, half, outerArguments(half), innerArguments)
generateCategorical2PassFinalKernel(batched_categorical_final_2pass, bfloat, outerArguments(bfloat), innerArguments)

#undef outerArguments
#undef innerArguments
#undef generateCategorical2PassFinalKernel

#undef BLOCK_SIZE
#undef GRAIN_SIZE 