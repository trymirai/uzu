#include <metal_stdlib>
#include <metal_simdgroup>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../generated/ring.h"
#include "../generated/trie.h"
#include "mask.h"

#define TOTAL_BLOCKS_COUNT 32
#define SIMDGROUPS_PER_THREADGROUP 8

using namespace uzu::ring;
using namespace uzu::trie;

// GQA-aware two-pass attention (pass 1).
// Fixed 256-thread threadgroups (8 simdgroups) with flat query-head mapping.
// Each simdgroup independently handles one query head and derives its KV head
// via integer division. Simdgroups sharing a KV head benefit from L1 cache.
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
    const ThreadContext thread_context,
    const uint head_group_idx GROUPS(num_heads.div_ceil(SIMDGROUPS_PER_THREADGROUP)),
    const uint q_seq_idx GROUPS(suffix_length),
    const uint block_idx GROUPS(TOTAL_BLOCKS_COUNT),
    const uint tid THREADS(SIMDGROUPS_PER_THREADGROUP * METAL_SIMD_SIZE)
) {
  constexpr uint elements_per_thread = HEAD_DIM / METAL_SIMD_SIZE;

  typedef float U;

  // Flat query head mapping — fixed hardware shape, dynamic logical mapping
  const uint head_idx = head_group_idx * SIMDGROUPS_PER_THREADGROUP +
                        thread_context.threadgroup_index;
  if (head_idx >= num_heads)
    return;

  const uint kv_head_idx = head_idx / gqa_factor;
  const uint o_offset = q_seq_idx * num_heads + head_idx;
  const uint q_offset = head_idx * suffix_length + q_seq_idx;

  const uint prefix_length = sequence_length - suffix_length;
  const uint suffix_position =
      is_kv_cache_ring ? uint(ring_params.ring_length) : prefix_length;
  const uint query_position = is_trie ? suffix_position + trie[q_seq_idx].height
                                      : suffix_position + q_seq_idx;

  thread U q[elements_per_thread];
  thread U o[elements_per_thread] = {0};

  queries += q_offset * HEAD_DIM +
             thread_context.simdgroup_index * elements_per_thread;
  keys += kv_head_idx * k_head_stride + block_idx * k_seq_stride +
          thread_context.simdgroup_index * elements_per_thread;
  values += kv_head_idx * v_head_stride + block_idx * v_seq_stride +
            thread_context.simdgroup_index * elements_per_thread;
  out += o_offset * TOTAL_BLOCKS_COUNT * HEAD_DIM + block_idx * HEAD_DIM +
         thread_context.simdgroup_index * elements_per_thread;
  sums += o_offset * TOTAL_BLOCKS_COUNT + block_idx;
  maxs += o_offset * TOTAL_BLOCKS_COUNT + block_idx;

  for (uint i = 0; i < elements_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  U max_score = -1e9;
  U sum_exp_score = 0;
  if (has_sinks && block_idx == 0) {
    max_score = static_cast<U>(sinks[head_idx]);
    sum_exp_score = 1;
  }

  // For each key position
  for (uint i = block_idx; i < sequence_length; i += TOTAL_BLOCKS_COUNT) {
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
      // Compute score
      U score = 0;
      for (uint j = 0; j < elements_per_thread; j++) {
        score += q[j] * static_cast<U>(keys[j]);
      }
      score = simd_sum(score);

      // Online softmax update
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      for (uint j = 0; j < elements_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * static_cast<U>(values[j]);
      }
    }

    keys += TOTAL_BLOCKS_COUNT * k_seq_stride;
    values += TOTAL_BLOCKS_COUNT * v_seq_stride;
  }

  // Write partial output
  if (thread_context.simdgroup_index == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }
  for (uint i = 0; i < elements_per_thread; i++) {
    out[i] = o[i];
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
    const ThreadContext thread_context,
    const uint head_idx GROUPS(num_heads),
    const uint q_seq_idx GROUPS(suffix_length),
    const uint tid THREADS(1024)
) {
  constexpr uint elements_per_thread = HEAD_DIM / METAL_SIMD_SIZE;

  typedef float U;

  const uint o_offset = q_seq_idx * num_heads + head_idx;

  sums += o_offset * TOTAL_BLOCKS_COUNT;
  maxs += o_offset * TOTAL_BLOCKS_COUNT;

  // Find global max across all blocks (each lane handles strided blocks)
  U max_score = -1e9;
  for (uint b = thread_context.simdgroup_index; b < TOTAL_BLOCKS_COUNT;
       b += METAL_SIMD_SIZE) {
    max_score = max(max_score, maxs[b]);
  }
  U new_max = simd_max(max_score);

  // Find global sum across all blocks
  U sum_exp = 0;
  for (uint b = thread_context.simdgroup_index; b < TOTAL_BLOCKS_COUNT;
       b += METAL_SIMD_SIZE) {
    sum_exp += sums[b] * fast::exp(maxs[b] - new_max);
  }
  U sum_exp_score = simd_sum(sum_exp);

  // Combine partial outputs — each lane reads a different block,
  // simd_sum reduces across blocks directly (no shared memory needed)
  thread U o[elements_per_thread] = {0};

  for (uint chunk = 0; chunk < TOTAL_BLOCKS_COUNT; chunk += METAL_SIMD_SIZE) {
    uint block_idx = chunk + thread_context.simdgroup_index;
    U factor = (block_idx < TOTAL_BLOCKS_COUNT)
                   ? fast::exp(maxs[block_idx] - new_max)
                   : 0;

    const device float* block_partials =
        partials + o_offset * TOTAL_BLOCKS_COUNT * HEAD_DIM +
        block_idx * HEAD_DIM +
        thread_context.threadgroup_index * elements_per_thread;

    for (uint i = 0; i < elements_per_thread; i++) {
      U val = (block_idx < TOTAL_BLOCKS_COUNT) ? block_partials[i] * factor : 0;
      o[i] += simd_sum(val);
    }
  }

  // Write the output
  if (thread_context.simdgroup_index == 0) {
    out += o_offset * HEAD_DIM +
           thread_context.threadgroup_index * elements_per_thread;
    for (uint i = 0; i < elements_per_thread; i++) {
      out[i] = static_cast<T>(o[i] / sum_exp_score);
    }
  }
}
