#include <metal_stdlib>
#include <metal_simdgroup>
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../generated/ring.h"
#include "../generated/trie.h"
#include "mask.h"

#define SEQUENCE_BLOCK_SIZE 32
#define HEAD_BLOCK_SIZE 32

using namespace uzu::ring;
using namespace uzu::trie;

template <typename T, uint HEAD_DIM>
VARIANTS(T, float, half, bfloat)
VARIANTS(HEAD_DIM, 64, 128, 256, 512)
PUBLIC KERNEL(AttentionSinglePass)(
    const device T* queries,
    const device T* keys,
    const device T* values,
    device T* out,
    const constant uint& gqa_factor,
    const constant uint& sequence_length,
    const constant uint& k_head_stride,
    const constant uint& k_seq_stride,
    const constant uint& v_head_stride,
    const constant uint& v_seq_stride,
    const constant RingParams& ring_params OPTIONAL(is_kv_cache_ring),
    const constant float& scale,
    const device TrieNode* trie OPTIONAL(is_trie),
    const constant uint& sliding_window_size OPTIONAL(is_sliding_window),
    const device float* sinks OPTIONAL(has_sinks),
    const constant uint& num_heads,
    const constant uint& suffix_length,
    threadgroup float shared_max_scores[SEQUENCE_BLOCK_SIZE * HEAD_BLOCK_SIZE],
    threadgroup float shared_sum_exp_scores[SEQUENCE_BLOCK_SIZE * HEAD_BLOCK_SIZE],
    threadgroup float shared_outputs[SEQUENCE_BLOCK_SIZE * HEAD_BLOCK_SIZE],
    const bool has_sinks SPECIALIZE,
    const bool is_kv_cache_ring SPECIALIZE,
    const bool is_causal SPECIALIZE,
    const bool is_trie SPECIALIZE,
    const bool is_sliding_window SPECIALIZE,
    const ThreadContext thread_context,
    const uint head_idx GROUPS(num_heads),
    const uint q_seq_idx GROUPS(suffix_length),
    const uint tid THREADS(1024)
) {
  constexpr bool query_transposed = false;

  constexpr uint value_dim = HEAD_DIM;
  constexpr uint qk_elements_per_thread = HEAD_DIM / HEAD_BLOCK_SIZE;
  constexpr uint value_elements_per_thread = value_dim / HEAD_BLOCK_SIZE;
  uint inner_k_stride = SEQUENCE_BLOCK_SIZE * int(k_seq_stride);
  uint inner_v_stride = SEQUENCE_BLOCK_SIZE * int(v_seq_stride);

  typedef float U;

  thread U q[qk_elements_per_thread];
  thread U k[qk_elements_per_thread];
  thread U o[value_elements_per_thread];

  const uint kv_head_idx = head_idx / gqa_factor;
  const uint o_offset = q_seq_idx * num_heads + head_idx;
  const uint q_offset = query_transposed ? num_heads * q_seq_idx + head_idx
                                         : head_idx * suffix_length + q_seq_idx;

  const uint prefix_length = sequence_length - suffix_length;

  const uint suffix_position =
      is_kv_cache_ring ? uint(ring_params.ring_length) : prefix_length;

  const uint query_position = is_trie ? suffix_position + trie[q_seq_idx].height
                                      : suffix_position + q_seq_idx;

  queries += q_offset * HEAD_DIM +
             thread_context.simdgroup_index * qk_elements_per_thread;
  keys += kv_head_idx * k_head_stride +
          thread_context.threadgroup_index * k_seq_stride +
          thread_context.simdgroup_index * qk_elements_per_thread;
  values += kv_head_idx * v_head_stride +
            thread_context.threadgroup_index * v_seq_stride +
            thread_context.simdgroup_index * value_elements_per_thread;

  out += o_offset * value_dim +
         thread_context.threadgroup_index * value_elements_per_thread;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < qk_elements_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < value_elements_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;
  if (has_sinks && thread_context.threadgroup_index == 0) {
    const int num_q_heads = static_cast<int>(num_heads);
    int q_head_idx = head_idx % num_q_heads;
    max_score = static_cast<U>(sinks[q_head_idx]);
    sum_exp_score = 1;
  }

  // For each key
  for (uint i = thread_context.threadgroup_index; i < sequence_length;
       i += SEQUENCE_BLOCK_SIZE) {
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
      for (int j = 0; j < value_elements_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }

    // Move the pointers to the next kv
    keys += inner_k_stride;
    values += inner_v_stride;
  }

  // Each thread has a partial part of the output so we need to combine them.
  if (thread_context.simdgroup_index == 0) {
    shared_max_scores[thread_context.threadgroup_index] = max_score;
    shared_sum_exp_scores[thread_context.threadgroup_index] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = shared_max_scores[thread_context.simdgroup_index];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score =
      simd_sum(shared_sum_exp_scores[thread_context.simdgroup_index] * factor);

  // Now we need to aggregate all the outputs
  for (int i = 0; i < value_elements_per_thread; i++) {
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
    for (int i = 0; i < value_elements_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}
