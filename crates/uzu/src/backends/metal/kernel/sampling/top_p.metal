#include <metal_stdlib>
#include <metal_atomic>
#include "../definitions.metal"
#include "../rng.metal"
using namespace metal;

#define BLOCK_SIZE 1024
#define GRAIN_SIZE 4

struct SamplingTempStorage {
    _atomic<uint> sampled_id;
    _atomic<int>  last_valid_id;
    float         chunk_prob_sum;
    float         min_val;
    float         max_val;
    float         pivot0_mass_broadcast;
    float         pivot1_mass_broadcast;
};

inline void process_probability_chunk(
        uint           chunk_idx, uint vocab_size,
        float          prob_threshold,
        float          random_sample,
        thread const float (&prob_vec)[GRAIN_SIZE],
        thread float  &cumulative_prob, 
        threadgroup    SamplingTempStorage& ts,
        uint           local_id,
        threadgroup    float* shared_sum,
        threadgroup    int*  shared_max)
{
    bool   valid[GRAIN_SIZE];
    float  filtered_probs[GRAIN_SIZE];
    float  cumulative_distribution[GRAIN_SIZE];

    /* build masked probability vector & validity bits */
    for (uint j=0;j<GRAIN_SIZE;++j) {
        bool keep_condition = prob_vec[j] > prob_threshold;
        filtered_probs[j] = keep_condition ? prob_vec[j] : 0.f;
        uint global_token_id = (chunk_idx*BLOCK_SIZE + local_id)*GRAIN_SIZE + j;
        valid[j] = keep_condition && global_token_id < vocab_size;
    }

    /* 1. block-reduce sum using optimized raking reduction */
    float local_sum = filtered_probs[0] + filtered_probs[1] + 
                      filtered_probs[2] + filtered_probs[3];
    float block_sum = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(local_sum, shared_sum, local_id);
    
    if (local_id == 0) ts.chunk_prob_sum = block_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    block_sum = ts.chunk_prob_sum;

    /* 2. check if candidate is in this chunk */
    if (cumulative_prob + block_sum > random_sample) {
        /* compute inclusive scan within this thread's GRAIN_SIZE elements */
        thread_prefix_inclusive_sum<GRAIN_SIZE>(filtered_probs);
        for (uint j = 0; j < GRAIN_SIZE; ++j) {
            cumulative_distribution[j] = filtered_probs[j];
        }
        
        /* get prefix sum from previous threads using optimized raking scan */
        float thread_prefix = threadgroup_raking_prefix_exclusive_sum<BLOCK_SIZE>(local_sum, shared_sum, local_id);
        
        /* add global prefix to local inclusive scan */
        for (uint j = 0; j < GRAIN_SIZE; ++j) {
            cumulative_distribution[j] += thread_prefix + cumulative_prob;
        }

        /* find elements that cross the sample threshold */
        bool exceeds_sample[GRAIN_SIZE];
        bool is_first_match[GRAIN_SIZE];
        
        for (uint j = 0; j < GRAIN_SIZE; ++j) {
            exceeds_sample[j] = (cumulative_distribution[j] > random_sample) && valid[j];
        }
        
        /* find first match elements (first occurrence in each run) */
        is_first_match[0] = exceeds_sample[0];
        for (uint j = 1; j < GRAIN_SIZE; ++j) {
            is_first_match[j] = exceeds_sample[j] && !exceeds_sample[j-1];
        }

        /* atomic update for first qualifying element */
        for (uint j = 0; j < GRAIN_SIZE; ++j) {
            if (is_first_match[j]) {
                uint global_token_id = (chunk_idx*BLOCK_SIZE + local_id)*GRAIN_SIZE + j;
                atomic_fetch_min_explicit(&ts.sampled_id, global_token_id, memory_order_relaxed);
            }
        }
    }

    /* 3. update last_valid_id using optimized raking reduction */
    int local_last_valid = -1;  // sentinel smaller than any real token id
    for (uint j = 0; j < GRAIN_SIZE; ++j) {
        if (valid[j]) {
            local_last_valid = int((chunk_idx*BLOCK_SIZE + local_id)*GRAIN_SIZE + j);
        }
    }
    
    int max_valid_token = threadgroup_cooperative_reduce_max<BLOCK_SIZE>(local_last_valid, shared_max, local_id);
    if (local_id == 0) {
        atomic_fetch_max_explicit(&ts.last_valid_id, max_valid_token, memory_order_relaxed);
    }

    /* 4. update running cumulative probability */
    cumulative_prob += block_sum;
}

template<typename T>
void topp_sampling_kernel_impl(
    const device T* logits,
    device uint*    out_ids,
    constant uint&  batch,
    constant uint&  vocab,
    constant float& top_p,
    constant uint64_t& seed,
    threadgroup SamplingTempStorage& ts,
    threadgroup float* s_max,
    threadgroup float* s_sum,
    threadgroup int*  s_last_idx,
    threadgroup float* shared_pivot0_reduction,
    threadgroup float* shared_pivot1_reduction,
    uint batch_idx,
    uint lid)
{
    if (batch_idx>=batch) return;

    const uint stride_elems   = BLOCK_SIZE * GRAIN_SIZE;
    const uint num_vec_chunks = (vocab + stride_elems - 1) / stride_elems;
    const uint row_off        = batch_idx * vocab;

    PhiloxState rng;
    uint64_t philox_offset = batch_idx;
    philox_init(&rng, seed, philox_offset);

    /* ── init shared with proper barriers */
    if (lid==0){
        atomic_store_explicit(&ts.sampled_id, vocab, memory_order_relaxed);
        atomic_store_explicit(&ts.last_valid_id, -1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* ── max-logit reduction using optimized raking reduction */
    float local_max = -INFINITY;
    for(uint i = lid; i < vocab; i += BLOCK_SIZE) {
        local_max = max(local_max, float(logits[row_off + i]));
    }
    float max_logit = threadgroup_cooperative_reduce_max<BLOCK_SIZE>(local_max, s_max, lid);

    /* ── sum-exp reduction using optimized raking reduction */
    float local_sum = 0.f;
    for (uint i = lid; i < vocab; i += BLOCK_SIZE) {
        local_sum += exp(float(logits[row_off + i]) - max_logit);
    }
    float sum_exp = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(local_sum, s_sum, lid);

    /* ── rejection-sampling loop */
    float low = 0.0f, high = 1.0f, interval_mass = 1.0f;
    uint sampled = vocab;

    do {
        /* Proper atomic initialization with barriers */
        if (lid == 0) {
            atomic_store_explicit(&ts.sampled_id   , vocab, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float sample_target = uniform_float(&rng) * interval_mass;
        float running_prob_sum = 0.f;

        /* ---------- PASS A ---------- */
        for (uint vec_i = 0; vec_i < num_vec_chunks; ++vec_i) {
            /* load vec */
            float prob_vec[GRAIN_SIZE];
            for (uint j = 0; j < GRAIN_SIZE; ++j) {
                uint gid = (vec_i * BLOCK_SIZE + lid) * GRAIN_SIZE + j;
                prob_vec[j] = (gid < vocab) ? exp(float(logits[row_off + gid]) - max_logit) / sum_exp : 0.f;
            }

            process_probability_chunk(
                vec_i, vocab,
                low,
                sample_target,
                prob_vec,
                running_prob_sum,
                ts, lid,
                s_sum,
                s_last_idx);

            if (running_prob_sum > sample_target) break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        sampled = atomic_load_explicit(&ts.sampled_id, memory_order_relaxed);
        if (sampled == vocab) {
            int last_valid = atomic_load_explicit(&ts.last_valid_id, memory_order_relaxed);
            if (last_valid >= 0) {    // only cast when we have a real index
                sampled = uint(last_valid);
            }
        }

        /* If we still don't have a valid token, restart the loop */
        if (sampled >= vocab) {
            continue;    // go back to the top of the rejection loop
        }

        /* ---------- PASS B ---------- */
        float sampled_p = exp(float(logits[row_off + sampled]) - max_logit) / sum_exp;
        float pivot0 = sampled_p;
        float pivot1 = (pivot0 + high) / 2.0f;

        float pivot0_mass = 0.0f, pivot1_mass = 0.0f;
        
        for (uint vec_i = 0; vec_i < num_vec_chunks; ++vec_i) {
            float local_pivot0_sum = 0.f, local_pivot1_sum = 0.f;
            for (uint j = 0; j < GRAIN_SIZE; ++j) {
                uint gid = (vec_i * BLOCK_SIZE + lid) * GRAIN_SIZE + j;
                if (gid < vocab) {
                    float p = exp(float(logits[row_off + gid]) - max_logit) / sum_exp;
                    if (p > pivot0) local_pivot0_sum += p;
                    if (p > pivot1) local_pivot1_sum += p;
                }
            }
        
            float block_pivot0_sum = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(local_pivot0_sum, shared_pivot0_reduction, lid);
            float block_pivot1_sum = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(local_pivot1_sum, shared_pivot1_reduction, lid);
            
            if (lid == 0) {
                pivot0_mass += block_pivot0_sum;
                pivot1_mass += block_pivot1_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        /* FIX: Broadcast pivot masses to all threads */
        if (lid == 0) {
            ts.pivot0_mass_broadcast = pivot0_mass;
            ts.pivot1_mass_broadcast = pivot1_mass;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        pivot0_mass = ts.pivot0_mass_broadcast;
        pivot1_mass = ts.pivot1_mass_broadcast;

        /* acceptance logic */
        if (pivot0_mass < top_p) break;
        if (pivot1_mass < top_p) {
            low = pivot0; high = pivot1; interval_mass = pivot0_mass;
        } else {
            low = pivot1; interval_mass = pivot1_mass;
        }
    } while(low < high);

    if (lid == 0) {
        out_ids[batch_idx] = sampled;
    }
}

// Generate top-p kernels using macro approach
#define outerArguments(T) \
(device const T* logits_data [[ buffer(0) ]], \
device uint* sampled_tokens [[ buffer(1) ]], \
constant uint& batch_size [[ buffer(2) ]], \
constant uint& vocab_size [[ buffer(3) ]], \
constant float& top_p [[ buffer(4) ]], \
constant uint64_t& seed [[ buffer(5) ]], \
uint batch_idx [[ threadgroup_position_in_grid ]], \
uint local_id [[ thread_position_in_threadgroup ]])

#define innerArguments \
(logits_data, sampled_tokens, batch_size, vocab_size, top_p, seed, temp_storage, shared_max, shared_sum, shared_last_idx, shared_pivot0_reduction, shared_pivot1_reduction, batch_idx, local_id)

#define generateToppKernel(functionName, T, outerArgs, innerArgs) \
kernel void functionName##_##T outerArgs { \
    threadgroup SamplingTempStorage temp_storage; \
    threadgroup float shared_max[BLOCK_SIZE]; \
    threadgroup float shared_sum[BLOCK_SIZE]; \
    threadgroup int  shared_last_idx[BLOCK_SIZE]; \
    threadgroup float shared_pivot0_reduction[BLOCK_SIZE]; \
    threadgroup float shared_pivot1_reduction[BLOCK_SIZE]; \
    topp_sampling_kernel_impl<T> innerArgs; \
}

generateToppKernel(batched_topp_main, float, outerArguments(float), innerArguments)
generateToppKernel(batched_topp_main, half, outerArguments(half), innerArguments)
generateToppKernel(batched_topp_main, bfloat, outerArguments(bfloat), innerArguments)

#undef outerArguments
#undef innerArguments
#undef generateToppKernel

#undef BLOCK_SIZE
#undef GRAIN_SIZE

