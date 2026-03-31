#include <metal_stdlib>
#include <metal_simdgroup>
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../generated/ring.h"
#include "../generated/trie.h"
#include "mask.h"

#define HEAD_BLOCK_SIZE 32
#define SEQUENCE_BLOCK_SIZE_1 8
#define SEQUENCE_BLOCK_SIZE_2 32
#define TOTAL_BLOCKS_COUNT 32

using namespace uzu::ring;
using namespace uzu::trie;

template <typename T, uint HEAD_DIM>
VARIANTS(T, float, half, bfloat)
VARIANTS(HEAD_DIM, 64, 128, 256)
PUBLIC KERNEL(AttentionTwoPass1)(
    const device T* queries,
    const device T* keys,
    const device T* values,
    device float* out,
    device float* sums,
    device float* maxs,
    const constant uint& gqa_factor,
    const constant uint& sequence_length,
    const constant uint& k_head_stride,
    const constant uint& k_seq_stride,
    const constant uint& v_head_stride,
    const constant uint& v_seq_stride,
    const constant RingParams& ring_params OPTIONAL(is_kv_cache_ring),
    const constant float& scale,
    const constant uint& num_heads,
    const constant uint& suffix_length,
    const device TrieNode* trie OPTIONAL(is_trie),
    const constant uint& sliding_window_size OPTIONAL(is_sliding_window),
    const device float* sinks OPTIONAL(has_sinks),
    const bool has_sinks SPECIALIZE,
    const bool is_kv_cache_ring SPECIALIZE,
    const bool is_causal SPECIALIZE,
    const bool is_trie SPECIALIZE,
    const bool is_sliding_window SPECIALIZE,
    threadgroup float shared_max_scores[SEQUENCE_BLOCK_SIZE_1],
    threadgroup float shared_sum_exp_scores[SEQUENCE_BLOCK_SIZE_1],
    threadgroup float shared_outputs[SEQUENCE_BLOCK_SIZE_1 * HEAD_BLOCK_SIZE],
    const ThreadContext thread_context,
    const uint head_idx GROUPS(num_heads),
    const uint q_seq_idx GROUPS(suffix_length),
    const uint block_idx GROUPS(TOTAL_BLOCKS_COUNT),
    const uint tid THREADS(256)
) {
  constexpr bool query_transposed = false;
  constexpr uint value_dim = HEAD_DIM;
  constexpr uint qk_elements_per_thread = HEAD_DIM / HEAD_BLOCK_SIZE;
  constexpr uint value_elements_per_thread = value_dim / HEAD_BLOCK_SIZE;
  uint inner_k_stride = SEQUENCE_BLOCK_SIZE_1 * k_seq_stride;
  uint inner_v_stride = SEQUENCE_BLOCK_SIZE_1 * v_seq_stride;

  typedef float U;

  thread U q[qk_elements_per_thread];
  thread U k[qk_elements_per_thread];
  thread U o[value_elements_per_thread];

  const uint o_offset = q_seq_idx * num_heads + head_idx; // Our custom layout
  const uint q_offset =
      query_transposed
          ? num_heads * q_seq_idx + head_idx
          : head_idx * suffix_length + q_seq_idx; // Consistent with single-pass
  const uint kv_head_idx = head_idx / gqa_factor;

  const uint prefix_length = sequence_length - suffix_length;

  const uint suffix_position =
      is_kv_cache_ring ? uint(ring_params.ring_length) : prefix_length;

  const uint query_position = is_trie ? suffix_position + trie[q_seq_idx].height
                                      : suffix_position + q_seq_idx;

  queries += q_offset * HEAD_DIM +
             thread_context.simdgroup_index * qk_elements_per_thread;
  keys +=
      kv_head_idx * k_head_stride +
      (block_idx * SEQUENCE_BLOCK_SIZE_1 + thread_context.threadgroup_index) *
          k_seq_stride +
      thread_context.simdgroup_index * qk_elements_per_thread;
  values +=
      kv_head_idx * v_head_stride +
      (block_idx * SEQUENCE_BLOCK_SIZE_1 + thread_context.threadgroup_index) *
          v_seq_stride +
      thread_context.simdgroup_index * value_elements_per_thread;
  out += o_offset * TOTAL_BLOCKS_COUNT * value_dim + block_idx * value_dim +
         thread_context.simdgroup_index * value_elements_per_thread;
  sums += o_offset * TOTAL_BLOCKS_COUNT + block_idx;
  maxs += o_offset * TOTAL_BLOCKS_COUNT + block_idx;

  // Read the query and 0 the output accumulator
  for (uint i = 0; i < qk_elements_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (uint i = 0; i < value_elements_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -1e9;
  U sum_exp_score = 0;
  if (has_sinks && block_idx == 0 && thread_context.threadgroup_index == 0) {
    const uint num_q_heads = num_heads;
    int q_head_idx = head_idx % num_q_heads;
    max_score = static_cast<U>(sinks[q_head_idx]);
    sum_exp_score = 1;
  }

  // For each key
  for (uint i =
           block_idx * SEQUENCE_BLOCK_SIZE_1 + thread_context.threadgroup_index;
       i < sequence_length;
       i += TOTAL_BLOCKS_COUNT * SEQUENCE_BLOCK_SIZE_1) {
    if (should_use_key(
            ring_params,
            trie,
            sliding_window_size,
            q_seq_idx,
            prefix_length,
            suffix_position,
            query_position,
            i,
            is_kv_cache_ring,
            is_causal,
            is_trie,
            is_sliding_window
        )) {
      // Read the key
      for (uint j = 0; j < qk_elements_per_thread; j++) {
        k[j] = keys[j];
      }

      // Compute the i-th score
      U score = 0;
      for (uint j = 0; j < qk_elements_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (uint j = 0; j < value_elements_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }

    // Move the pointers to the next kv
    keys += TOTAL_BLOCKS_COUNT * inner_k_stride;
    values += TOTAL_BLOCKS_COUNT * inner_v_stride;
  }

  // Each thread has a partial part of the output so we need to combine them.
  if (thread_context.simdgroup_index == 0) {
    shared_max_scores[thread_context.threadgroup_index] = max_score;
    shared_sum_exp_scores[thread_context.threadgroup_index] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = (thread_context.simdgroup_index < SEQUENCE_BLOCK_SIZE_1)
                  ? shared_max_scores[thread_context.simdgroup_index]
                  : -1e9;
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = (thread_context.simdgroup_index < SEQUENCE_BLOCK_SIZE_1)
                      ? shared_sum_exp_scores[thread_context.simdgroup_index]
                      : 0;
  sum_exp_score = simd_sum(sum_exp_score * factor);

  // Write the sum and new max
  if (thread_context.threadgroup_index == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = new_max;
  }

  // Now we need to aggregate all the outputs
  for (uint i = 0; i < value_elements_per_thread; i++) {
    shared_outputs
        [thread_context.simdgroup_index * SEQUENCE_BLOCK_SIZE_1 +
         thread_context.threadgroup_index] =
            o[i] *
            fast::exp(
                shared_max_scores[thread_context.threadgroup_index] - new_max
            );
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // And write the output
    if (thread_context.threadgroup_index == 0) {
      U output = shared_outputs
          [thread_context.simdgroup_index * SEQUENCE_BLOCK_SIZE_1];
      for (uint j = 1; j < SEQUENCE_BLOCK_SIZE_1; j++) {
        output += shared_outputs
            [thread_context.simdgroup_index * SEQUENCE_BLOCK_SIZE_1 + j];
      }
      out[i] = static_cast<T>(output);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <typename T, uint HEAD_DIM>
VARIANTS(T, float, half, bfloat)
VARIANTS(HEAD_DIM, 64, 128, 256)
PUBLIC KERNEL(AttentionTwoPass2)(
    const device float* partials,
    const device float* sums,
    const device float* maxs,
    device T* out,
    const constant uint& num_heads,
    const constant uint& suffix_length,
    threadgroup float shared_outputs[SEQUENCE_BLOCK_SIZE_2 * HEAD_BLOCK_SIZE],
    const ThreadContext thread_context,
    const uint head_idx GROUPS(num_heads),
    const uint q_seq_idx GROUPS(suffix_length),
    const uint tid THREADS(1024)
) {
  constexpr uint elements_per_thread = HEAD_DIM / HEAD_BLOCK_SIZE;

  typedef float U;
  thread U o[elements_per_thread];

  const uint o_offset = q_seq_idx * num_heads + head_idx; // Our custom layout

  partials += o_offset * TOTAL_BLOCKS_COUNT * HEAD_DIM +
              thread_context.threadgroup_index * HEAD_DIM +
              thread_context.simdgroup_index * elements_per_thread;
  sums += o_offset * TOTAL_BLOCKS_COUNT;
  maxs += o_offset * TOTAL_BLOCKS_COUNT;
  out +=
      o_offset * HEAD_DIM + thread_context.threadgroup_index *
                                elements_per_thread; // Our custom output layout

  // First everybody reads the max and sum_exp
  U max_score = maxs[thread_context.simdgroup_index];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  U sum_exp_score = simd_sum(sums[thread_context.simdgroup_index] * factor);

  // Now read the block into registers and then use shared memory to transpose
  // it
  for (uint i = 0; i < elements_per_thread; i++) {
    o[i] = partials[i];
  }
  for (uint i = 0; i < elements_per_thread; i++) {
    shared_outputs
        [thread_context.simdgroup_index * HEAD_BLOCK_SIZE +
         thread_context.threadgroup_index] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(
               shared_outputs
                   [thread_context.threadgroup_index * HEAD_BLOCK_SIZE +
                    thread_context.simdgroup_index] *
               factor
           ) /
           sum_exp_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (thread_context.simdgroup_index == 0) {
    for (uint i = 0; i < elements_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}
