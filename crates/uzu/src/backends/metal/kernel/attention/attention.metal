#include <metal_stdlib>
#include <metal_simdgroup>
#include "../definitions.metal"

using namespace metal;

constant bool has_mask [[function_constant(20)]];
constant bool query_transposed [[function_constant(21)]];
constant bool do_causal [[function_constant(22)]];
constant bool bool_mask [[function_constant(23)]];
constant bool float_mask [[function_constant(24)]];
constant bool has_sinks [[function_constant(25)]];

template <typename T, int head_dim, int value_dim = head_dim>
void attention_single_pass_impl(
    const device T* queries,
    const device T* keys,
    const device T* values,
    device T* out,
    const constant int& gqa_factor,
    const constant int& sequence_length,
    const constant int& k_head_stride,
    const constant int& k_seq_stride,
    const constant int& v_head_stride,
    const constant int& v_seq_stride,
    const constant float& scale,
    const device bool* bmask,
    const device T* fmask,
    const constant int& mask_kv_seq_stride,
    const constant int& mask_q_seq_stride,
    const constant int& mask_head_stride,
    const device float* sinks,
    uint3 tid, // threadgroup position in grid
    uint3 tpg, // threadgroups per grid
    uint simd_gid, // simdgroup index in threadgroup
    uint simd_lid, // thread index in simdgroup
    threadgroup float* shared_max_scores,
    threadgroup float* shared_sum_exp_scores,
    threadgroup float* shared_outputs
) {
    constexpr int sequence_block_size = 32;
    constexpr int head_block_size = 32;
    constexpr int qk_elements_per_thread = head_dim / head_block_size;
    constexpr int value_elements_per_thread = value_dim / head_block_size;
    int inner_k_stride = sequence_block_size * int(k_seq_stride);
    int inner_v_stride = sequence_block_size * int(v_seq_stride);

    typedef float U;

    thread U q[qk_elements_per_thread];
    thread U k[qk_elements_per_thread];
    thread U o[value_elements_per_thread];

    const int head_idx = tid.x;
    const int q_seq_idx = tid.y;
    const int kv_head_idx = head_idx / gqa_factor;
    const int o_offset = q_seq_idx * tpg.x + head_idx;
    const int q_offset = query_transposed ? tpg.x * q_seq_idx + head_idx : head_idx * tpg.y + q_seq_idx;
    
    queries += q_offset * head_dim + simd_lid * qk_elements_per_thread;
    keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride + simd_lid * qk_elements_per_thread;
    values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride + simd_lid * value_elements_per_thread;
    
    if (bool_mask) {
        bmask += head_idx * mask_head_stride + simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
    }
    if (float_mask) {
        fmask += head_idx * mask_head_stride + simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
    }

    out += o_offset * value_dim + simd_gid * value_elements_per_thread;

    // Read the query and 0 the output accumulator
    for (int i = 0; i < qk_elements_per_thread; i++) {
        q[i] = static_cast<U>(scale) * queries[i];
    }
    for (int i = 0; i < value_elements_per_thread; i++) {
        o[i] = 0;
    }

    U max_score = -INFINITY;
    U sum_exp_score = 0;
    if (has_sinks && simd_gid == 0) {
        const int num_q_heads = static_cast<int>(tpg.x);
        int q_head_idx = head_idx % num_q_heads;
        max_score = static_cast<U>(sinks[q_head_idx]);
        sum_exp_score = 1;
    }

    // For each key
    for (int i = simd_gid; i < sequence_length; i += sequence_block_size) {
        bool use_key = true;
        if (do_causal) {
            use_key = i <= (sequence_length - int(tpg.y) + int(q_seq_idx));
        } else if (bool_mask) {
            use_key = bmask[0];
        }
        
        if (use_key) {
            // Read the key
            for (int j = 0; j < qk_elements_per_thread; j++) {
                k[j] = keys[j];
            }

            // Compute the i-th score
            U score = 0;
            for (int j = 0; j < qk_elements_per_thread; j++) {
                score += q[j] * k[j];
            }
            score = simd_sum(score);
            if (float_mask) {
                score += max(-1e9f, static_cast<U>(fmask[0]));
            }

            // Update the accumulators
            U new_max = max(max_score, score);
            U factor = fast::exp(max_score - new_max);
            U exp_score = fast::exp(score - new_max);

            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;

            // Update the output accumulator
            for (int j = 0; j < value_elements_per_thread; j++) {
                o[j] = o[j] * factor + exp_score * values[j];
            }
        }

        // Move the pointers to the next kv
        keys += inner_k_stride;
        values += inner_v_stride;
        if (bool_mask) {
            bmask += sequence_block_size * mask_kv_seq_stride;
        }
        if (float_mask) {
            fmask += sequence_block_size * mask_kv_seq_stride;
        }
    }

    // Each thread has a partial part of the output so we need to combine them.
    if (simd_lid == 0) {
        shared_max_scores[simd_gid] = max_score;
        shared_sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_score = shared_max_scores[simd_lid];
    U new_max = simd_max(max_score);
    U factor = fast::exp(max_score - new_max);
    sum_exp_score = simd_sum(shared_sum_exp_scores[simd_lid] * factor);

    // Now we need to aggregate all the outputs
    for (int i = 0; i < value_elements_per_thread; i++) {
        shared_outputs[simd_lid * head_block_size + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(shared_outputs[simd_gid * head_block_size + simd_lid] * factor) / sum_exp_score;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // And write the output
    if (simd_lid == 0) {
        for (int i = 0; i < value_elements_per_thread; i++) {
            out[i] = static_cast<T>(o[i]);
        }
    }
}

template <typename T, int head_dim, int value_dim = head_dim>
void attention_2pass_1_impl(
    const device T* queries,
    const device T* keys,
    const device T* values,
    device float* out,
    device float* sums,
    device float* maxs,
    const constant int& gqa_factor,
    const constant int& sequence_length,
    const constant int& k_head_stride,
    const constant int& k_seq_stride,
    const constant int& v_head_stride,
    const constant int& v_seq_stride,
    const constant float& scale,
    const device bool* bmask,
    const device T* fmask,
    const constant int& mask_kv_seq_stride,
    const constant int& mask_q_seq_stride,
    const constant int& mask_head_stride,
    const device float* sinks,
    uint3 tid, // threadgroup position in grid
    uint3 tpg, // threadgroups per grid
    uint simd_gid, // simdgroup index in threadgroup
    uint simd_lid, // thread index in simdgroup
    threadgroup float* shared_max_scores,
    threadgroup float* shared_sum_exp_scores,
    threadgroup float* shared_outputs
) {
    constexpr int sequence_block_size = 8;
    constexpr int head_block_size = 32;
    constexpr int qk_elements_per_thread = head_dim / head_block_size;
    constexpr int value_elements_per_thread = value_dim / head_block_size;
    int inner_k_stride = sequence_block_size * int(k_seq_stride);
    int inner_v_stride = sequence_block_size * int(v_seq_stride);
    constexpr int total_blocks_count = 32;

    typedef float U;

    thread U q[qk_elements_per_thread];
    thread U k[qk_elements_per_thread];
    thread U o[value_elements_per_thread];

    const int block_idx = tid.z;
    const int head_idx = tid.x;
    const int q_seq_idx = tid.y;
    const int o_offset = q_seq_idx * tpg.x + head_idx; // Our custom layout
    const int q_offset = query_transposed ? tpg.x * q_seq_idx + head_idx : head_idx * tpg.y + q_seq_idx; // Consistent with single-pass
    const int kv_head_idx = head_idx / gqa_factor;

    queries += q_offset * head_dim + simd_lid * qk_elements_per_thread;
    keys += kv_head_idx * k_head_stride + (block_idx * sequence_block_size + simd_gid) * k_seq_stride + simd_lid * qk_elements_per_thread;
    values += kv_head_idx * v_head_stride + (block_idx * sequence_block_size + simd_gid) * v_seq_stride + simd_lid * value_elements_per_thread;
    out += o_offset * total_blocks_count * value_dim + block_idx * value_dim + simd_lid * value_elements_per_thread;
    
    if (bool_mask) {
        bmask += head_idx * mask_head_stride + (block_idx * sequence_block_size + simd_gid) * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
    }
    if (float_mask) {
        fmask += head_idx * mask_head_stride + (block_idx * sequence_block_size + simd_gid) * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
    }
    sums += o_offset * total_blocks_count + block_idx;
    maxs += o_offset * total_blocks_count + block_idx;

    // Read the query and 0 the output accumulator
    for (int i = 0; i < qk_elements_per_thread; i++) {
        q[i] = static_cast<U>(scale) * queries[i];
    }
    for (int i = 0; i < value_elements_per_thread; i++) {
        o[i] = 0;
    }

    U max_score = -1e9;
    U sum_exp_score = 0;
    if (has_sinks && block_idx == 0 && simd_gid == 0) {
        const int num_q_heads = static_cast<int>(tpg.x);
        int q_head_idx = head_idx % num_q_heads;
        max_score = static_cast<U>(sinks[q_head_idx]);
        sum_exp_score = 1;
    }

    // For each key
    for (int i = block_idx * sequence_block_size + simd_gid; i < sequence_length; i += total_blocks_count * sequence_block_size) {
        bool use_key = true;
        if (do_causal) {
            use_key = i <= (sequence_length - int(tpg.y) + int(q_seq_idx));
        } else if (bool_mask) {
            use_key = bmask[0];
        }
        
        if (use_key) {
            // Read the key
            for (int j = 0; j < qk_elements_per_thread; j++) {
                k[j] = keys[j];
            }

            // Compute the i-th score
            U score = 0;
            for (int j = 0; j < qk_elements_per_thread; j++) {
                score += q[j] * k[j];
            }
            score = simd_sum(score);
            if (float_mask) {
                score += max(-1e9f, static_cast<U>(fmask[0]));
            }

            // Update the accumulators
            U new_max = max(max_score, score);
            U factor = fast::exp(max_score - new_max);
            U exp_score = fast::exp(score - new_max);

            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;

            // Update the output accumulator
            for (int j = 0; j < value_elements_per_thread; j++) {
                o[j] = o[j] * factor + exp_score * values[j];
            }
        }

        // Move the pointers to the next kv
        keys += total_blocks_count * inner_k_stride;
        values += total_blocks_count * inner_v_stride;
        if (bool_mask) {
            bmask += sequence_block_size * total_blocks_count * mask_kv_seq_stride;
        }
        if (float_mask) {
            fmask += sequence_block_size * total_blocks_count * mask_kv_seq_stride;
        }
    }

    // Each thread has a partial part of the output so we need to combine them.
    if (simd_lid == 0) {
        shared_max_scores[simd_gid] = max_score;
        shared_sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_score = (simd_lid < sequence_block_size) ? shared_max_scores[simd_lid] : -1e9;
    U new_max = simd_max(max_score);
    U factor = fast::exp(max_score - new_max);
    sum_exp_score = (simd_lid < sequence_block_size) ? shared_sum_exp_scores[simd_lid] : 0;
    sum_exp_score = simd_sum(sum_exp_score * factor);

    // Write the sum and new max
    if (simd_gid == 0) {
        sums[0] = sum_exp_score;
        maxs[0] = new_max;
    }

    // Now we need to aggregate all the outputs
    for (int i = 0; i < value_elements_per_thread; i++) {
        shared_outputs[simd_lid * sequence_block_size + simd_gid] = o[i] * fast::exp(shared_max_scores[simd_gid] - new_max);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // And write the output
        if (simd_gid == 0) {
            U output = shared_outputs[simd_lid * sequence_block_size];
            for (int j = 1; j < sequence_block_size; j++) {
                output += shared_outputs[simd_lid * sequence_block_size + j];
            }
            out[i] = static_cast<T>(output);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template <typename T, int head_dim>
void attention_2pass_2_impl(
    const device float* partials,
    const device float* sums,
    const device float* maxs,
    device T* out,
    uint3 tid,
    uint3 tpg,
    uint simd_gid,
    uint simd_lid,
    threadgroup float* shared_outputs
) {
    constexpr int sequence_block_size = 32;
    constexpr int head_block_size = 32;
    constexpr int elements_per_thread = head_dim / head_block_size;
    constexpr int total_blocks_count = 32;

    typedef float U;

    thread U o[elements_per_thread];

    const int head_idx = tid.x;
    const int q_seq_idx = tid.y;
    const int o_offset = q_seq_idx * tpg.x + head_idx; // Our custom layout
    const int q_offset = query_transposed ? tpg.x * q_seq_idx + head_idx : head_idx * tpg.y + q_seq_idx; // Consistent with single-pass
    
    partials += o_offset * total_blocks_count * head_dim + simd_gid * head_dim + simd_lid * elements_per_thread;
    sums += o_offset * total_blocks_count;
    maxs += o_offset * total_blocks_count;
    out += o_offset * head_dim + simd_gid * elements_per_thread; // Our custom output layout

    // First everybody reads the max and sum_exp
    U max_score = maxs[simd_lid];
    U new_max = simd_max(max_score);
    U factor = fast::exp(max_score - new_max);
    U sum_exp_score = simd_sum(sums[simd_lid] * factor);

    // Now read the block into registers and then use shared memory to transpose it
    for (int i = 0; i < elements_per_thread; i++) {
        o[i] = partials[i];
    }
    for (int i = 0; i < elements_per_thread; i++) {
        shared_outputs[simd_lid * head_block_size + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(shared_outputs[simd_gid * head_block_size + simd_lid] * factor) / sum_exp_score;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // And write the output
    if (simd_lid == 0) {
        for (int i = 0; i < elements_per_thread; i++) {
            out[i] = static_cast<T>(o[i]);
        }
    }
}

template <typename T, int head_dim, int value_dim = head_dim>
void attention_single_pass(
    const device T* queries,
    const device T* keys,
    const device T* values,
    device T* out,
    const constant int& gqa_factor,
    const constant int& sequence_length,
    const constant int& k_head_stride,
    const constant int& k_seq_stride,
    const constant int& v_head_stride,
    const constant int& v_seq_stride,
    const constant float& scale,
    const device bool* bmask,
    const device T* fmask,
    const constant int& mask_kv_seq_stride,
    const constant int& mask_q_seq_stride,
    const constant int& mask_head_stride,
    const device float* sinks,
    uint3 tid,
    uint3 tpg,
    uint simd_gid,
    uint simd_lid,
    threadgroup float* shared_max_scores,
    threadgroup float* shared_sum_exp_scores,
    threadgroup float* shared_outputs
) {
    attention_single_pass_impl<T, head_dim, value_dim>(
        queries, keys, values, out, gqa_factor, sequence_length,
        k_head_stride, k_seq_stride, v_head_stride, v_seq_stride,
        scale, bmask, fmask, mask_kv_seq_stride, mask_q_seq_stride, mask_head_stride,
        sinks,
        tid, tpg, simd_gid, simd_lid,
        shared_max_scores, shared_sum_exp_scores, shared_outputs
    );
}

template <typename T, int head_dim, int value_dim = head_dim>
void attention_2pass_1(
    const device T* queries,
    const device T* keys,
    const device T* values,
    device float* out,
    device float* sums,
    device float* maxs,
    const constant int& gqa_factor,
    const constant int& sequence_length,
    const constant int& k_head_stride,
    const constant int& k_seq_stride,
    const constant int& v_head_stride,
    const constant int& v_seq_stride,
    const constant float& scale,
    const device bool* bmask,
    const device T* fmask,
    const constant int& mask_kv_seq_stride,
    const constant int& mask_q_seq_stride,
    const constant int& mask_head_stride,
    const device float* sinks,
    uint3 tid,
    uint3 tpg,
    uint simd_gid,
    uint simd_lid,
    threadgroup float* shared_max_scores,
    threadgroup float* shared_sum_exp_scores,
    threadgroup float* shared_outputs
) {
    attention_2pass_1_impl<T, head_dim, value_dim>(
        queries, keys, values, out, sums, maxs, gqa_factor, sequence_length,
        k_head_stride, k_seq_stride, v_head_stride, v_seq_stride,
        scale, bmask, fmask, mask_kv_seq_stride, mask_q_seq_stride, mask_head_stride,
        sinks,
        tid, tpg, simd_gid, simd_lid,
        shared_max_scores, shared_sum_exp_scores, shared_outputs
    );
}

template <typename T, int head_dim>
void attention_2pass_2(
    const device float* partials,
    const device float* sums,
    const device float* maxs,
    device T* out,
    uint3 tid,
    uint3 tpg,
    uint simd_gid,
    uint simd_lid,
    threadgroup float* shared_outputs
) {
    attention_2pass_2_impl<T, head_dim>(
        partials, sums, maxs, out, tid, tpg, simd_gid, simd_lid, shared_outputs
    );
}

#define outerArguments(T)                                          \
(const device T* queries       [[ buffer(0) ]],                   \
 const device T* keys          [[ buffer(1) ]],                   \
 const device T* values        [[ buffer(2) ]],                   \
 device T* out                 [[ buffer(3) ]],                   \
 const constant int& gqa_factor [[ buffer(4) ]],                  \
 const constant int& sequence_length [[ buffer(5) ]],             \
 const constant int& k_head_stride [[ buffer(6) ]],               \
 const constant int& k_seq_stride [[ buffer(7) ]],                \
 const constant int& v_head_stride [[ buffer(8) ]],               \
 const constant int& v_seq_stride [[ buffer(9) ]],                \
 const constant float& scale   [[ buffer(10) ]],                  \
 const device bool* bmask      [[ buffer(11), function_constant(bool_mask) ]], \
 const device T* fmask         [[ buffer(12), function_constant(float_mask) ]], \
 const constant int& mask_kv_seq_stride [[ buffer(13), function_constant(has_mask) ]], \
 const constant int& mask_q_seq_stride [[ buffer(14), function_constant(has_mask) ]], \
 const constant int& mask_head_stride [[ buffer(15), function_constant(has_mask) ]], \
 const device float* sinks     [[ buffer(16), function_constant(has_sinks) ]], \
 uint3 tid                     [[ threadgroup_position_in_grid ]], \
 uint3 tpg                     [[ threadgroups_per_grid ]],       \
 uint simd_gid                 [[ simdgroup_index_in_threadgroup ]], \
 uint simd_lid                 [[ thread_index_in_simdgroup ]])   \

#define innerArguments                                              \
(queries, keys, values, out, gqa_factor, sequence_length, k_head_stride, k_seq_stride, \
 v_head_stride, v_seq_stride, scale, bmask, fmask, mask_kv_seq_stride, \
 mask_q_seq_stride, mask_head_stride, sinks, tid, tpg, simd_gid, simd_lid, \
 shared_max_scores, shared_sum_exp_scores, shared_outputs) \

// Generate single-pass kernels for different head dimensions
#define GENERATE_SINGLE_PASS_KERNELS(head_dim_value) \
[[max_total_threads_per_threadgroup(1024)]] kernel void attention_single_pass_half_##head_dim_value outerArguments(half) { \
    constexpr int sequence_block_size = 32; \
    constexpr int head_block_size = 32; \
    threadgroup float shared_max_scores[sequence_block_size]; \
    threadgroup float shared_sum_exp_scores[sequence_block_size]; \
    threadgroup float shared_outputs[sequence_block_size * head_block_size]; \
    attention_single_pass<half, head_dim_value> innerArguments; \
} \
[[max_total_threads_per_threadgroup(1024)]] kernel void attention_single_pass_bfloat_##head_dim_value outerArguments(bfloat) { \
    constexpr int sequence_block_size = 32; \
    constexpr int head_block_size = 32; \
    threadgroup float shared_max_scores[sequence_block_size]; \
    threadgroup float shared_sum_exp_scores[sequence_block_size]; \
    threadgroup float shared_outputs[sequence_block_size * head_block_size]; \
    attention_single_pass<bfloat, head_dim_value> innerArguments; \
} \
[[max_total_threads_per_threadgroup(1024)]] kernel void attention_single_pass_float_##head_dim_value outerArguments(float) { \
    constexpr int sequence_block_size = 32; \
    constexpr int head_block_size = 32; \
    threadgroup float shared_max_scores[sequence_block_size]; \
    threadgroup float shared_sum_exp_scores[sequence_block_size]; \
    threadgroup float shared_outputs[sequence_block_size * head_block_size]; \
    attention_single_pass<float, head_dim_value> innerArguments; \
}

GENERATE_SINGLE_PASS_KERNELS(64)
GENERATE_SINGLE_PASS_KERNELS(128)
GENERATE_SINGLE_PASS_KERNELS(256)

#undef outerArguments
#undef innerArguments

#define outerArguments(T)                                          \
(const device T* queries       [[ buffer(0) ]],                   \
 const device T* keys          [[ buffer(1) ]],                   \
 const device T* values        [[ buffer(2) ]],                   \
 device float* out             [[ buffer(3) ]],                   \
 device float* sums            [[ buffer(4) ]],                   \
 device float* maxs            [[ buffer(5) ]],                   \
 const constant int& gqa_factor [[ buffer(6) ]],                  \
 const constant int& sequence_length [[ buffer(7) ]],             \
 const constant int& k_head_stride [[ buffer(8) ]],               \
 const constant int& k_seq_stride [[ buffer(9) ]],                \
 const constant int& v_head_stride [[ buffer(10) ]],              \
 const constant int& v_seq_stride [[ buffer(11) ]],               \
 const constant float& scale   [[ buffer(12) ]],                  \
 const device bool* bmask      [[ buffer(13), function_constant(bool_mask) ]], \
 const device T* fmask         [[ buffer(14), function_constant(float_mask) ]], \
 const constant int& mask_kv_seq_stride [[ buffer(15), function_constant(has_mask) ]], \
 const constant int& mask_q_seq_stride [[ buffer(16), function_constant(has_mask) ]], \
 const constant int& mask_head_stride [[ buffer(17), function_constant(has_mask) ]], \
 const device float* sinks     [[ buffer(18), function_constant(has_sinks) ]], \
 uint3 tid                     [[ threadgroup_position_in_grid ]], \
 uint3 tpg                     [[ threadgroups_per_grid ]],       \
 uint simd_gid                 [[ simdgroup_index_in_threadgroup ]], \
 uint simd_lid                 [[ thread_index_in_simdgroup ]])   \

#define innerArguments                                              \
(queries, keys, values, out, sums, maxs, gqa_factor, sequence_length, k_head_stride, k_seq_stride, \
 v_head_stride, v_seq_stride, scale, bmask, fmask, mask_kv_seq_stride, \
 mask_q_seq_stride, mask_head_stride, sinks, tid, tpg, simd_gid, simd_lid, \
 shared_max_scores, shared_sum_exp_scores, shared_outputs) \

// Generate 2-pass Pass 1 kernels for different head dimensions
#define GENERATE_2PASS_1_KERNELS(head_dim_value) \
[[max_total_threads_per_threadgroup(256)]] kernel void attention_2pass_1_half_##head_dim_value outerArguments(half) { \
    constexpr int sequence_block_size = 8; \
    constexpr int head_block_size = 32; \
    threadgroup float shared_max_scores[sequence_block_size]; \
    threadgroup float shared_sum_exp_scores[sequence_block_size]; \
    threadgroup float shared_outputs[sequence_block_size * head_block_size]; \
    attention_2pass_1<half, head_dim_value> innerArguments; \
} \
[[max_total_threads_per_threadgroup(256)]] kernel void attention_2pass_1_bfloat_##head_dim_value outerArguments(bfloat) { \
    constexpr int sequence_block_size = 8; \
    constexpr int head_block_size = 32; \
    threadgroup float shared_max_scores[sequence_block_size]; \
    threadgroup float shared_sum_exp_scores[sequence_block_size]; \
    threadgroup float shared_outputs[sequence_block_size * head_block_size]; \
    attention_2pass_1<bfloat, head_dim_value> innerArguments; \
} \
[[max_total_threads_per_threadgroup(256)]] kernel void attention_2pass_1_float_##head_dim_value outerArguments(float) { \
    constexpr int sequence_block_size = 8; \
    constexpr int head_block_size = 32; \
    threadgroup float shared_max_scores[sequence_block_size]; \
    threadgroup float shared_sum_exp_scores[sequence_block_size]; \
    threadgroup float shared_outputs[sequence_block_size * head_block_size]; \
    attention_2pass_1<float, head_dim_value> innerArguments; \
}

GENERATE_2PASS_1_KERNELS(64)
GENERATE_2PASS_1_KERNELS(128)
GENERATE_2PASS_1_KERNELS(256)

#undef outerArguments
#undef innerArguments

#define outerArguments(T)                                          \
(const device float* partials  [[ buffer(0) ]],                   \
 const device float* sums      [[ buffer(1) ]],                   \
 const device float* maxs      [[ buffer(2) ]],                   \
 device T* out                 [[ buffer(3) ]],                   \
 uint3 tid                     [[ threadgroup_position_in_grid ]], \
 uint3 tpg                     [[ threadgroups_per_grid ]],       \
 uint simd_gid                 [[ simdgroup_index_in_threadgroup ]], \
 uint simd_lid                 [[ thread_index_in_simdgroup ]])   \

#define innerArguments                                              \
(partials, sums, maxs, out, tid, tpg, simd_gid, simd_lid, shared_outputs) \

// Generate 2-pass Pass 2 kernels for different head dimensions
#define GENERATE_2PASS_2_KERNELS(head_dim_value) \
[[max_total_threads_per_threadgroup(1024)]] kernel void attention_2pass_2_half_##head_dim_value outerArguments(half) { \
    constexpr int sequence_block_size = 32; \
    constexpr int head_block_size = 32; \
    threadgroup float shared_outputs[sequence_block_size * head_block_size]; \
    attention_2pass_2<half, head_dim_value> innerArguments; \
} \
[[max_total_threads_per_threadgroup(1024)]] kernel void attention_2pass_2_float_##head_dim_value outerArguments(float) { \
    constexpr int sequence_block_size = 32; \
    constexpr int head_block_size = 32; \
    threadgroup float shared_outputs[sequence_block_size * head_block_size]; \
    attention_2pass_2<float, head_dim_value> innerArguments; \
}

GENERATE_2PASS_2_KERNELS(64)
GENERATE_2PASS_2_KERNELS(128)
GENERATE_2PASS_2_KERNELS(256)

#undef outerArguments
#undef innerArguments

template <typename T>
void update_kv_cache(
    const device T* rotated_keys,        // [num_groups, suffix_length, head_dim]
    const device T* qkv,                 // [suffix_length, (num_heads + 2*num_groups) * head_dim]
    device T* key_cache,                 // [num_groups, max_sequence_length, head_dim]
    device T* value_cache,               // [num_groups, max_sequence_length, head_dim]
    const constant int& num_groups,
    const constant int& num_heads,
    const constant int& head_dim,
    const constant int& suffix_length,
    const constant int& prefix_segment_length,
    const constant int& max_sequence_length,
    uint3 position                       // x: group_idx, y: token_idx, z: dim_idx
) {
    const uint groupIndex = position.x;
    const uint tokenIndex = position.y;
    const uint dimIndex = position.z;
    
    if (groupIndex >= num_groups || tokenIndex >= suffix_length || dimIndex >= head_dim) {
        return;
    }
    
    TensorView3D<const T> rotatedKeysTensorView = TensorView3D<const T>(rotated_keys, num_groups, suffix_length, head_dim);
    TensorView3D<T> keyCacheTensorView = TensorView3D<T>(key_cache, num_groups, max_sequence_length, head_dim);
    TensorView3D<T> valueCacheTensorView = TensorView3D<T>(value_cache, num_groups, max_sequence_length, head_dim);
    
    // Copy rotated key to cache
    keyCacheTensorView(groupIndex, prefix_segment_length + tokenIndex, dimIndex) = rotatedKeysTensorView(groupIndex, tokenIndex, dimIndex);
    
    // Update value cache (only first thread in each token processes values to avoid redundant work)
    if (dimIndex == 0) {
        const uint totalQueryDim = num_heads * head_dim;
        const uint totalKeyValueDim = num_groups * head_dim;
        
        const int qkvStride = totalQueryDim + 2 * totalKeyValueDim;
        TensorView2D<const T> qkvTensorView = TensorView2D<const T>(qkv, suffix_length, qkvStride);
        
        // Extract values from QKV tensor
        // Values start at offset: total_query_dim + total_key_value_dim
        for (uint d = 0; d < head_dim; d++) {
            const uint valueOffset = totalQueryDim + totalKeyValueDim + groupIndex * head_dim + d;
            valueCacheTensorView(groupIndex, prefix_segment_length + tokenIndex, d) = qkvTensorView(tokenIndex, valueOffset);
        }
    }
}

#define outerArguments(T)                                          \
(const device T* rotated_keys    [[ buffer(0) ]],                 \
 const device T* qkv             [[ buffer(1) ]],                 \
 device T* key_cache             [[ buffer(2) ]],                 \
 device T* value_cache           [[ buffer(3) ]],                 \
 const constant int& num_groups  [[ buffer(4) ]],                 \
 const constant int& num_heads   [[ buffer(5) ]],                 \
 const constant int& head_dim    [[ buffer(6) ]],                 \
 const constant int& suffix_length [[ buffer(7) ]],               \
 const constant int& prefix_segment_length [[ buffer(8) ]],       \
 const constant int& max_sequence_length [[ buffer(9) ]],         \
 uint3 position                  [[ thread_position_in_grid ]])   \

#define innerArguments                                             \
(rotated_keys, qkv, key_cache, value_cache, num_groups, num_heads, \
 head_dim, suffix_length, prefix_segment_length, max_sequence_length, position) \

generateKernels(update_kv_cache)

#undef outerArguments
#undef innerArguments
