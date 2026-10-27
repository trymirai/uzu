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

template <uint BLOCK_THREADS>
VARIANTS(BLOCK_THREADS, 256, 1024)
PUBLIC KERNEL(WeaverSelect)(
    const device bfloat* residual_logits OPTIONAL(mode != WeaverSelectionMode::Frontier),
    const device uint64_t* depth_seeds OPTIONAL(mode != WeaverSelectionMode::Frontier),
    device uint* frontier,
    device uint* packed_tree,
    device uint* slot_ancestors OPTIONAL(mode != WeaverSelectionMode::Children),
    device uint* node_token_ids OPTIONAL(mode != WeaverSelectionMode::Children),
    device uint* node_metadata,
    device uint* node_ancestor_indices OPTIONAL(mode != WeaverSelectionMode::Children),
    device uint* node_valid,
    const device uint* candidate_pool_ids,
    const device float* candidate_pool_logits,
    device uint* node_candidate_ids,
    device float* node_candidate_logits,
    constant uint& children_rows,
    constant uint& children_from_pool,
    constant uint& candidates,
    constant uint& expand_width,
    constant uint& vocab_size,
    constant uint& frontier_capacity,
    constant uint& tree_slot_count,
    constant uint& node_count,
    constant uint& batch_start_slot,
    constant uint& ancestor_stride,
    constant uint& lookahead_count,
    constant uint& candidate_depth_count,
    constant float& prune_noise_scale OPTIONAL(has_prune_noise),
    const WeaverSelectionMode mode SPECIALIZE,
    const bool has_prune_noise SPECIALIZE,
    threadgroup float shared_float[TOP_CHILDREN_SIMDGROUPS * (BLOCK_THREADS / TOP_CHILDREN_THREADS)],
    threadgroup uint4 reduce_storage[FRONTIER_SELECT_SIMDGROUPS * (BLOCK_THREADS / TOP_CHILDREN_THREADS)],
    threadgroup float statistics[2 * (BLOCK_THREADS / TOP_CHILDREN_THREADS)],
    threadgroup uint top_winners[2 * (BLOCK_THREADS / TOP_CHILDREN_THREADS)],
    threadgroup uint winner_slot[FRONTIER_MAX_WIDTH],
    threadgroup uint node_candidate_depth[FRONTIER_MAX_WIDTH],
    const ThreadContext global_context,
    const uint group_index GROUPS(if mode == WeaverSelectionMode::Children { children_rows } else {1}),
    const uint global_lid THREADS(BLOCK_THREADS)
) {
  if (mode != WeaverSelectionMode::Frontier && candidates != 0 && candidates <= CANDIDATES_MAX && expand_width != 0 &&
      expand_width <= candidates && frontier_capacity != 0 && tree_slot_count != 0) {
    const uint rows = children_rows;
    const uint block = global_lid / TOP_CHILDREN_THREADS;
    const uint row = mode == WeaverSelectionMode::Fused ? block : group_index;
    const uint lid = global_lid % TOP_CHILDREN_THREADS;
    ThreadContext thread_context = global_context;
    thread_context.simdgroup_index %= TOP_CHILDREN_SIMDGROUPS;
    threadgroup float* reduce_float = shared_float + block * TOP_CHILDREN_SIMDGROUPS;
    threadgroup uint* reduce_score =
        reinterpret_cast<threadgroup uint*>(reduce_storage) + block * TOP_CHILDREN_SIMDGROUPS * 3;
    threadgroup uint* reduce_token = reduce_score + TOP_CHILDREN_SIMDGROUPS;
    threadgroup uint* reduce_index = reduce_token + TOP_CHILDREN_SIMDGROUPS;
    threadgroup float& logit_max = statistics[block * 2];
    threadgroup float& log_sum = statistics[block * 2 + 1];
    threadgroup uint& winner_token = top_winners[block * 2];
    threadgroup uint& winner_index = top_winners[block * 2 + 1];
    const device float* candidate_logits = children_from_pool != 0u ? candidate_pool_logits : node_candidate_logits;
    const device uint* candidate_ids = children_from_pool != 0u ? candidate_pool_ids : node_candidate_ids;
    const bool row_active = row < rows && node_valid[row] != 0u &&
                            node_metadata[uint(MetadataIdx::TreeSlot) * rows + row] < tree_slot_count;
    if (mode == WeaverSelectionMode::Children && !row_active) {
      return;
    }
    const uint tree_slot = row_active ? node_metadata[uint(MetadataIdx::TreeSlot) * rows + row] : 0u;
    const uint frontier_base = tree_slot * expand_width;

    const uint base = row * candidates;
    const uint depth = row_active ? node_metadata[uint(MetadataIdx::Depth) * rows + row] : 0u;
    const uint64_t seed = row_active ? depth_seeds[depth] : 0ul;
    const uint first_index = lid;
    const uint second_index = lid + TOP_CHILDREN_THREADS;
    const bool first_valid = row_active && first_index < candidates;
    const bool second_valid = row_active && second_index < candidates;
    const float first_logit =
        first_valid ? candidate_logits[base + first_index] + float(residual_logits[base + first_index]) : -INFINITY;
    const float second_logit =
        second_valid ? candidate_logits[base + second_index] + float(residual_logits[base + second_index]) : -INFINITY;
    const uint first_token = first_valid ? uint(candidate_ids[base + first_index]) : 0xffffffffu;
    const uint second_token = second_valid ? uint(candidate_ids[base + second_index]) : 0xffffffffu;
    const uint first_score =
        first_valid ? top_k_score_key(first_logit + gumbel_noise(seed, first_token, vocab_size)) : 0u;
    const uint second_score =
        second_valid ? top_k_score_key(second_logit + gumbel_noise(seed, second_token, vocab_size)) : 0u;
    bool first_active = first_valid;
    bool second_active = second_valid;

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

      if (row_active && lid == 0) {
        const uint slot = frontier_base + child;
        if (slot < frontier_capacity) {
          const float winner_logit =
              candidate_logits[base + winner_index] + float(residual_logits[base + winner_index]);
          const float model_logprob = winner_logit - log_sum;
          const float path_logprob =
              as_type<float>(packed_tree[uint(TreeIdx::PathLogprobBits) * tree_slot_count + tree_slot]);
          const float cumulative_logprob = path_logprob + model_logprob;
          const uint depth = packed_tree[uint(TreeIdx::Depth) * tree_slot_count + tree_slot] + 1u;
          frontier[uint(FrontierIdx::TokenId) * frontier_capacity + slot] = winner_token;
          frontier[uint(FrontierIdx::ParentSlot) * frontier_capacity + slot] = tree_slot;
          frontier[uint(FrontierIdx::Depth) * frontier_capacity + slot] = depth;
          frontier[uint(FrontierIdx::PathLogprobBits) * frontier_capacity + slot] = as_type<uint>(cumulative_logprob);
          if (!has_prune_noise) {
            frontier[uint(FrontierIdx::EdgeLogprobBits) * frontier_capacity + slot] = as_type<uint>(model_logprob);
          }
          frontier[uint(FrontierIdx::PathScoreKey) * frontier_capacity + slot] = top_k_score_key(cumulative_logprob);
          frontier[uint(FrontierIdx::Active) * frontier_capacity + slot] = 1u;
        }
      }
      if (has_prune_noise && (first_index == winner_index || second_index == winner_index)) {
        const uint slot = frontier_base + child;
        if (slot < frontier_capacity) {
          const float edge_logprob =
              (first_index == winner_index ? first_prune_logit : second_prune_logit) - prune_log_sum;
          frontier[uint(FrontierIdx::EdgeLogprobBits) * frontier_capacity + slot] = as_type<uint>(edge_logprob);
        }
      }
      first_active = first_active && first_index != winner_index;
      second_active = second_active && second_index != winner_index;
    }
  }
  if (mode != WeaverSelectionMode::Children) {
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    const ThreadContext thread_context = global_context;
    const uint lid = global_lid;
    const uint threads = BLOCK_THREADS;
    const uint groups = BLOCK_THREADS / METAL_SIMD_SIZE;
    const uint entries = FRONTIER_MAX_SLOTS / threads;
    threadgroup uint4* reduce = reduce_storage;

    if (frontier_capacity == 0 || frontier_capacity > FRONTIER_MAX_SLOTS || node_count == 0 ||
        node_count > FRONTIER_MAX_WIDTH || ancestor_stride == 0 || tree_slot_count == 0 ||
        batch_start_slot + node_count > tree_slot_count || candidate_depth_count == 0 || candidates == 0) {
      return;
    }

    bool entry_active[FRONTIER_MAX_SLOTS / BLOCK_THREADS];
    for (uint entry = 0; entry < entries; ++entry) {
      const uint slot = lid + entry * threads;
      entry_active[entry] =
          slot < frontier_capacity && frontier[uint(FrontierIdx::Active) * frontier_capacity + slot] != 0u;
    }

    for (uint child = 0; child < node_count; ++child) {
      uint4 local = uint4(0u, FRONTIER_NO_WINNER, FRONTIER_NO_WINNER, FRONTIER_NO_WINNER);
      for (uint entry = 0; entry < entries; ++entry) {
        const uint slot = lid + entry * threads;
        if (entry_active[entry]) {
          const uint depth = frontier[uint(FrontierIdx::Depth) * frontier_capacity + slot];
          const uint key = frontier[uint(FrontierIdx::PathScoreKey) * frontier_capacity + slot];
          const uint parent = frontier[uint(FrontierIdx::ParentSlot) * frontier_capacity + slot];
          const uint token = frontier[uint(FrontierIdx::TokenId) * frontier_capacity + slot];
          const uint packed_key = ((depth < lookahead_count ? 1u : 0u) << 31) | (key >> 1);
          if (packed_key > local.x ||
              (packed_key == local.x && (parent < local.y || (parent == local.y && token < local.z)))) {
            local = uint4(packed_key, parent, token, slot);
          }
        }
      }

      uint4 simd;
      simd.x = simd_max(local.x);
      simd.y = simd_min(local.x == simd.x ? local.y : FRONTIER_NO_WINNER);
      simd.z = simd_min(all(local.xy == simd.xy) ? local.z : FRONTIER_NO_WINNER);
      simd.w = simd_min(all(local.xyz == simd.xyz) ? local.w : FRONTIER_NO_WINNER);
      if (thread_context.simd_lane_id == 0) {
        reduce[thread_context.simdgroup_index] = simd;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (thread_context.simdgroup_index == 0) {
        const uint4 partial = thread_context.simd_lane_id < groups
                                  ? reduce[thread_context.simd_lane_id]
                                  : uint4(0u, FRONTIER_NO_WINNER, FRONTIER_NO_WINNER, FRONTIER_NO_WINNER);
        uint4 selected;
        selected.x = simd_max(partial.x);
        selected.y = simd_min(partial.x == selected.x ? partial.y : FRONTIER_NO_WINNER);
        selected.z = simd_min(all(partial.xy == selected.xy) ? partial.z : FRONTIER_NO_WINNER);
        selected.w = simd_min(all(partial.xyz == selected.xyz) ? partial.w : FRONTIER_NO_WINNER);
        if (thread_context.simd_lane_id == 0) {
          winner_slot[child] = selected.w;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const uint selected = winner_slot[child];
      for (uint entry = 0; entry < entries; ++entry) {
        entry_active[entry] = entry_active[entry] && (lid + entry * threads) != selected;
      }
    }

    if (lid < node_count) {
      const uint node = lid;
      const uint slot = winner_slot[node];
      const bool real = slot != FRONTIER_NO_WINNER;
      const uint tree_slot = batch_start_slot + node;

      const uint token = real ? frontier[uint(FrontierIdx::TokenId) * frontier_capacity + slot] : 0u;
      const uint parent =
          real ? frontier[uint(FrontierIdx::ParentSlot) * frontier_capacity + slot] : FRONTIER_NO_WINNER;
      const uint depth = real ? frontier[uint(FrontierIdx::Depth) * frontier_capacity + slot] : 0u;
      const uint cumulative_logprob =
          real ? frontier[uint(FrontierIdx::PathLogprobBits) * frontier_capacity + slot] : 0u;
      const uint logprob = real ? frontier[uint(FrontierIdx::EdgeLogprobBits) * frontier_capacity + slot] : 0u;

      packed_tree[uint(TreeIdx::TokenId) * tree_slot_count + tree_slot] = token;
      packed_tree[uint(TreeIdx::ParentSlot) * tree_slot_count + tree_slot] = parent;
      packed_tree[uint(TreeIdx::Depth) * tree_slot_count + tree_slot] = depth;
      packed_tree[uint(TreeIdx::PathLogprobBits) * tree_slot_count + tree_slot] = cumulative_logprob;
      packed_tree[uint(TreeIdx::EdgeLogprobBits) * tree_slot_count + tree_slot] = logprob;
      packed_tree[uint(TreeIdx::Valid) * tree_slot_count + tree_slot] = real ? 1u : 0u;

      if (real) {
        frontier[uint(FrontierIdx::Active) * frontier_capacity + slot] = 0u;
      }

      const uint parent_slot = real && parent < tree_slot_count ? parent : 0u;
      for (uint index = 0; index < ancestor_stride; ++index) {
        const uint ancestor =
            real && index + 1u <= depth
                ? (index + 1u == depth ? parent_slot : slot_ancestors[parent_slot * ancestor_stride + index])
                : 0u;
        slot_ancestors[tree_slot * ancestor_stride + index] = ancestor;
        node_ancestor_indices[node * ancestor_stride + index] = ancestor;
      }

      node_token_ids[node] = token;
      node_metadata[uint(MetadataIdx::Depth) * node_count + node] = depth < lookahead_count ? depth : PADDING_DEPTH;
      node_metadata[uint(MetadataIdx::AncestorCount) * node_count + node] = depth;
      node_metadata[uint(MetadataIdx::TreeSlot) * node_count + node] = tree_slot;
      node_valid[node] = real && depth < lookahead_count ? 1u : 0u;

      node_candidate_depth[node] = depth;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint node = 0; node < node_count; ++node) {
      if (node_candidate_depth[node] >= candidate_depth_count) {
        continue;
      }
      const uint source = node_candidate_depth[node] * candidates;
      const uint destination = node * candidates;
      for (uint candidate = lid; candidate < candidates; candidate += threads) {
        node_candidate_ids[destination + candidate] = candidate_pool_ids[source + candidate];
        node_candidate_logits[destination + candidate] = candidate_pool_logits[source + candidate];
      }
    }
  }
}
