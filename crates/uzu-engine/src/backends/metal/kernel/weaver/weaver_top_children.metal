#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"
#include "../common/top_k.h"
#include "../rng.h"
#include "weaver_frontier.h"

using namespace metal;

#define GUMBEL_THREADGROUP_SIZE 1024u
#define GUMBEL_WORDS_PER_OFFSET 4u

METAL_FUNC uint2 gumbel_revidx(uint logit_idx, uint vocab_size) {
  const uint thread_idx = logit_idx % GUMBEL_THREADGROUP_SIZE;
  const uint thread_offset = div_ceil(vocab_size, GUMBEL_THREADGROUP_SIZE * GUMBEL_WORDS_PER_OFFSET) * thread_idx;
  const uint block_idx = logit_idx / GUMBEL_THREADGROUP_SIZE;
  return uint2(thread_offset + block_idx / GUMBEL_WORDS_PER_OFFSET, block_idx % GUMBEL_WORDS_PER_OFFSET);
}

METAL_FUNC float gumbel_noise(uint64_t seed, uint logit_idx, uint vocab_size) {
  const uint2 offset_word = gumbel_revidx(logit_idx, vocab_size);
  PhiloxState rng;
  philox_init(&rng, seed, offset_word.x);
  const float uniform = float(rng.output[offset_word.y]) * (1.0f / 4294967296.0f);
  return -log(-log(uniform));
}

// The noise the target sampler adds for this token: the 24-bit uniform of `uniform_float` in rng.h keeps it finite.
METAL_FUNC float target_gumbel_noise(uint64_t seed, uint logit_idx, uint vocab_size) {
  const uint2 offset_word = gumbel_revidx(logit_idx, vocab_size);
  PhiloxState rng;
  philox_init(&rng, seed, offset_word.x);
  const float uniform = float(max(rng.output[offset_word.y] >> 8, 1u)) * (1.0f / 16777216.0f);
  return -log(-log(uniform));
}

METAL_FUNC bool weaver_better(uint score, uint token, uint index, uint best_score, uint best_token, uint best_index) {
  return score > best_score ||
         (score == best_score && (token < best_token || (token == best_token && index < best_index)));
}

// One threadgroup per node: picks its `expand_width` children from the candidate pool and inserts them into the
// frontier (slot parent * expand_width + child) with their path logprob and score key.
PUBLIC KERNEL(WeaverTopChildren)(
    const device bfloat* residual_logits,
    const device float* candidate_logits,
    const device uint* candidate_ids,
    const device uint64_t* depth_seeds,
    const device uint* node_metadata,
    const device uint* node_valid,
    const device uint* packed_tree,
    device uint* frontier,
    constant uint& rows,
    constant uint& candidates,
    constant uint& expansion_candidates,
    constant uint& expand_width,
    constant uint& vocab_size,
    constant uint& frontier_capacity,
    constant uint& tree_slot_count,
    constant float& prune_noise_scale OPTIONAL(has_prune_noise),
    const bool has_prune_noise SPECIALIZE,
    threadgroup float reduce_float[TOP_CHILDREN_SIMDGROUPS],
    threadgroup uint reduce_score[TOP_CHILDREN_SIMDGROUPS],
    threadgroup uint reduce_token[TOP_CHILDREN_SIMDGROUPS],
    threadgroup uint reduce_index[TOP_CHILDREN_SIMDGROUPS],
    threadgroup float& logit_max,
    threadgroup float& log_sum,
    threadgroup uint& winner_token,
    threadgroup uint& winner_index,
    const ThreadContext thread_context,
    const uint row GROUPS(rows),
    const uint lid THREADS(TOP_CHILDREN_THREADS)
) {
  if (candidates == 0 || candidates > CANDIDATES_MAX || expand_width == 0 || expand_width > expansion_candidates ||
      expansion_candidates > candidates ||
      frontier_capacity == 0 || tree_slot_count == 0) {
    return;
  }
  // Padding nodes expand nothing (uniform over the threadgroup).
  if (node_valid[row] == 0u) {
    return;
  }
  const uint parent = node_metadata[uint(MetadataIdx::TreeSlot) * rows + row];
  if (parent >= tree_slot_count) {
    return;
  }

  const uint base = row * candidates;
  const uint depth = node_metadata[uint(MetadataIdx::Depth) * rows + row];
  const uint64_t seed = depth_seeds[depth];
  const uint first_index = lid;
  const uint second_index = lid + TOP_CHILDREN_THREADS;
  const bool first_valid = first_index < candidates;
  const bool second_valid = second_index < candidates;
  const float first_logit =
      first_valid ? candidate_logits[base + first_index] + float(residual_logits[base + first_index]) : -INFINITY;
  const float second_logit =
      second_valid ? candidate_logits[base + second_index] + float(residual_logits[base + second_index]) : -INFINITY;
  const uint first_token = first_valid ? uint(candidate_ids[base + first_index]) : 0xffffffffu;
  const uint second_token = second_valid ? uint(candidate_ids[base + second_index]) : 0xffffffffu;
  // Expansion follows the target's own Gumbel noise so the tree covers what it will sample; greedy
  // verification (no prune noise) adds none, so the expansion must not either or the tree, and with it the
  // tree-verify rounding, would depend on the session seed.
  const float first_noise = has_prune_noise ? gumbel_noise(seed, first_token, vocab_size) : 0.0f;
  const float second_noise = has_prune_noise ? gumbel_noise(seed, second_token, vocab_size) : 0.0f;
  const uint first_score = first_valid ? top_k_score_key(first_logit + first_noise) : 0u;
  const uint second_score = second_valid ? top_k_score_key(second_logit + second_noise) : 0u;
  // Children come only from the pool's first `expansion_candidates` (the pool is sorted by draft logit).
  bool first_active = first_valid && first_index < expansion_candidates;
  bool second_active = second_valid && second_index < expansion_candidates;

  const float local_max = fmax(first_logit, second_logit);
  const float simd_maximum = simd_max(local_max);
  if (thread_context.simd_lane_id == 0) {
    reduce_float[thread_context.simdgroup_index] = simd_maximum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_context.simdgroup_index == 0) {
    const float group_value =
        thread_context.simd_lane_id < TOP_CHILDREN_SIMDGROUPS ? reduce_float[thread_context.simd_lane_id] : -INFINITY;
    const float maximum = simd_max(group_value);
    if (thread_context.simd_lane_id == 0) {
      logit_max = maximum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float local_sum =
      (first_valid ? exp(first_logit - logit_max) : 0.0f) + (second_valid ? exp(second_logit - logit_max) : 0.0f);
  const float simd_total = simd_sum(local_sum);
  if (thread_context.simd_lane_id == 0) {
    reduce_float[thread_context.simdgroup_index] = simd_total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_context.simdgroup_index == 0) {
    const float group_value =
        thread_context.simd_lane_id < TOP_CHILDREN_SIMDGROUPS ? reduce_float[thread_context.simd_lane_id] : 0.0f;
    const float total = simd_sum(group_value);
    if (thread_context.simd_lane_id == 0) {
      log_sum = log(total) + logit_max;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Final pruning weighs each edge by the pool softmax of logits plus the target's noise scaled by 1 / sigma.
  float first_prune_logit = -INFINITY;
  float second_prune_logit = -INFINITY;
  float prune_log_sum = 0.0f;
  if (has_prune_noise) {
    if (first_valid) {
      first_prune_logit = first_logit + prune_noise_scale * target_gumbel_noise(seed, first_token, vocab_size);
    }
    if (second_valid) {
      second_prune_logit = second_logit + prune_noise_scale * target_gumbel_noise(seed, second_token, vocab_size);
    }
    const float prune_max = threadgroup_cooperative_reduce<SimdReduceMax<float>, TOP_CHILDREN_THREADS>(
        fmax(first_prune_logit, second_prune_logit),
        reduce_float,
        thread_context
    );
    const float prune_total = threadgroup_cooperative_reduce<SimdReduceSum<float>, TOP_CHILDREN_THREADS>(
        (first_valid ? exp(first_prune_logit - prune_max) : 0.0f) +
            (second_valid ? exp(second_prune_logit - prune_max) : 0.0f),
        reduce_float,
        thread_context
    );
    prune_log_sum = log(prune_total) + prune_max;
  }

  for (uint child = 0; child < expand_width; ++child) {
    uint local_score = 0u;
    uint local_token = 0xffffffffu;
    uint local_index = 0xffffffffu;
    if (first_active) {
      local_score = first_score;
      local_token = first_token;
      local_index = first_index;
    }
    if (second_active &&
        weaver_better(second_score, second_token, second_index, local_score, local_token, local_index)) {
      local_score = second_score;
      local_token = second_token;
      local_index = second_index;
    }

    const uint simd_score = simd_max(local_score);
    const uint simd_token = simd_min(local_score == simd_score ? local_token : 0xffffffffu);
    const uint simd_index =
        simd_min(local_score == simd_score && local_token == simd_token ? local_index : 0xffffffffu);
    if (thread_context.simd_lane_id == 0) {
      reduce_score[thread_context.simdgroup_index] = simd_score;
      reduce_token[thread_context.simdgroup_index] = simd_token;
      reduce_index[thread_context.simdgroup_index] = simd_index;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (thread_context.simdgroup_index == 0) {
      const bool lane_valid = thread_context.simd_lane_id < TOP_CHILDREN_SIMDGROUPS;
      const uint group_score = lane_valid ? reduce_score[thread_context.simd_lane_id] : 0u;
      const uint selected_score = simd_max(group_score);
      const uint group_token =
          lane_valid && group_score == selected_score ? reduce_token[thread_context.simd_lane_id] : 0xffffffffu;
      const uint selected_token = simd_min(group_token);
      const uint group_index = lane_valid && group_score == selected_score && group_token == selected_token
                                   ? reduce_index[thread_context.simd_lane_id]
                                   : 0xffffffffu;
      const uint selected_index = simd_min(group_index);
      if (thread_context.simd_lane_id == 0) {
        winner_token = selected_token;
        winner_index = selected_index;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint slot = parent * expand_width + child;
    if (lid == 0 && slot < frontier_capacity) {
      const float winner_logit = candidate_logits[base + winner_index] + float(residual_logits[base + winner_index]);
      // `volatile` keeps the rounded logprob as the operand of the path sum (fast-math may not reassociate).
      volatile float logprob_rounded = winner_logit - log_sum;
      const float logprob = logprob_rounded;
      const float cumulative_logprob =
          as_type<float>(packed_tree[uint(TreeIdx::PathLogprobBits) * tree_slot_count + parent]) + logprob;
      frontier[uint(FrontierIdx::TokenId) * frontier_capacity + slot] = winner_token;
      frontier[uint(FrontierIdx::ParentSlot) * frontier_capacity + slot] = parent;
      frontier[uint(FrontierIdx::Depth) * frontier_capacity + slot] =
          packed_tree[uint(TreeIdx::Depth) * tree_slot_count + parent] + 1u;
      frontier[uint(FrontierIdx::PathLogprobBits) * frontier_capacity + slot] = as_type<uint>(cumulative_logprob);
      frontier[uint(FrontierIdx::PathScoreKey) * frontier_capacity + slot] = top_k_score_key(cumulative_logprob);
      frontier[uint(FrontierIdx::Active) * frontier_capacity + slot] = 1u;
      // The edge lane feeds only final pruning; without prune noise it keeps the model logprob.
      if (!has_prune_noise) {
        frontier[uint(FrontierIdx::EdgeLogprobBits) * frontier_capacity + slot] = as_type<uint>(logprob);
      }
    }
    if (has_prune_noise && slot < frontier_capacity && (first_index == winner_index || second_index == winner_index)) {
      frontier[uint(FrontierIdx::EdgeLogprobBits) * frontier_capacity + slot] =
          as_type<uint>((first_index == winner_index ? first_prune_logit : second_prune_logit) - prune_log_sum);
    }
    first_active = first_active && first_index != winner_index;
    second_active = second_active && second_index != winner_index;
  }
}
