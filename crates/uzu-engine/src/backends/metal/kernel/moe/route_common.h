#pragma once

#include <metal_stdlib>

#include "../common/defines.h"
#include "../common/top_k.h"
#include "../generated/moe.h"

using namespace metal;

namespace uzu {
namespace moe {

UZU_CONST uint ROUTE_EXPERTS_PER_LANE = MOE_MAX_EXPERTS / METAL_SIMD_SIZE;

METAL_FUNC float route_sigmoid(float x) { return 1.0f / (1.0f + precise::exp(-x)); }

// Orders like jax.lax.top_k: larger scores first, NaN below every number, +0.0 above -0.0.
METAL_FUNC uint route_key(float score) { return isnan(score) ? top_k_score_key(-INFINITY) : top_k_score_key(score); }

// Routing tables shared by every expert kernel. A route is one (token, pick) pair, numbered token * top_k + pick.
struct RouteTables {
  device uint* route_experts;  // [tokens * top_k]: the expert of each route
  device float* route_weights; // [tokens * top_k]: the weight of each route
  device uint* expert_counts;  // [experts]: routes per expert
  device uint* expert_routes;  // [experts, tokens]: the routes of each expert, in no particular order
  device uint* active_experts; // [1 + experts]: the count, then an entry (see active_entry) per expert with a
                               // route, in increasing expert order
};

// expert | count << MOE_ACTIVE_EXPERT_BITS | first_route << (MOE_ACTIVE_EXPERT_BITS + MOE_ACTIVE_COUNT_BITS), the
// first route being expert_routes[expert * tokens].
METAL_FUNC uint active_entry(uint expert, uint count, uint first_route) {
  return expert | (count << MOE_ACTIVE_EXPERT_BITS) | (first_route << (MOE_ACTIVE_EXPERT_BITS + MOE_ACTIVE_COUNT_BITS));
}

struct RouteParams {
  uint tokens;
  uint experts;
  uint top_k;
  float routed_scaling_factor;
  bool bias_selects; // the expert bias shifts the scores that select experts
  bool bias_weights; // the expert bias shifts the weights of the selected experts
};

// The experts one token selects, as lanes [0, top_k) of a simdgroup hold them: pick order, distinct experts.
struct TokenPicks {
  uint expert;  // valid in lanes [0, top_k)
  float weight; // normalized and scaled, valid in lanes [0, top_k)
};

// Picks one token's experts on one simdgroup: larger keys first, the lower expert on equal keys.
template <typename LogitLoader>
METAL_FUNC TokenPicks select_token_experts(
    LogitLoader logit_at,
    const device float* expert_biases,
    const RouteParams params,
    const uint token,
    const uint lane
) {
  // A key of 0 marks a missing expert: every score's key is larger (route_key never returns 0).
  uint keys[ROUTE_EXPERTS_PER_LANE];
  float weights[ROUTE_EXPERTS_PER_LANE];
  METAL_PRAGMA_UNROLL
  for (uint slot = 0; slot < ROUTE_EXPERTS_PER_LANE; ++slot) {
    const uint expert = lane + slot * METAL_SIMD_SIZE;
    keys[slot] = 0;
    weights[slot] = 0.0f;
    if (expert < params.experts) {
      const float score = route_sigmoid(logit_at(token * params.experts + expert));
      const float bias = expert_biases != nullptr ? expert_biases[expert] : 0.0f;
      keys[slot] = route_key(params.bias_selects ? score + bias : score);
      weights[slot] = params.bias_weights ? score + bias : score;
    }
  }

  TokenPicks picks = {0, 0.0f};
  float weight_sum = 0.0f;
  uint picked = 0;
  for (uint pick = 0; pick < params.top_k; ++pick) {
    uint best_key = 0;
    uint best_slot = 0;
    float best_weight = 0.0f;
    METAL_PRAGMA_UNROLL
    for (uint slot = 0; slot < ROUTE_EXPERTS_PER_LANE; ++slot) {
      // Within a lane, a later slot holds a larger index, so only a strictly larger key wins.
      if ((picked & (1u << slot)) == 0 && keys[slot] > best_key) {
        best_key = keys[slot];
        best_slot = slot;
        best_weight = weights[slot];
      }
    }
    const uint top_key = simd_max(best_key);
    const uint candidate = best_key == top_key ? lane + best_slot * METAL_SIMD_SIZE : 0xFFFFFFFFu;
    const uint best_expert = simd_min(candidate);
    const float weight = simd_shuffle(best_weight, ushort(best_expert % METAL_SIMD_SIZE));
    if (best_expert % METAL_SIMD_SIZE == lane) {
      picked |= 1u << (best_expert / METAL_SIMD_SIZE);
    }
    weight_sum += weight;
    if (lane == pick) {
      picks.expert = best_expert;
      picks.weight = weight;
    }
  }
  picks.weight = picks.weight / (weight_sum + 1e-6f) * params.routed_scaling_factor;
  return picks;
}

// Routing of a single token on simdgroup 0 alone: no threadgroup memory or barriers.
template <typename LogitLoader>
METAL_FUNC void route_single_token(
    LogitLoader logit_at,
    const device float* expert_biases,
    const RouteParams params,
    const RouteTables tables,
    const uint lane
) {
  const TokenPicks picks = select_token_experts(logit_at, expert_biases, params, 0, lane);
  const bool picked = lane < params.top_k;
  uint masks[ROUTE_EXPERTS_PER_LANE];
  METAL_PRAGMA_UNROLL
  for (uint word = 0; word < ROUTE_EXPERTS_PER_LANE; ++word) {
    masks[word] =
        simd_or(picked && picks.expert / METAL_SIMD_SIZE == word ? 1u << (picks.expert % METAL_SIMD_SIZE) : 0u);
  }
  METAL_PRAGMA_UNROLL
  for (uint slot = 0; slot < ROUTE_EXPERTS_PER_LANE; ++slot) {
    const uint expert = lane + slot * METAL_SIMD_SIZE;
    if (expert < params.experts) {
      tables.expert_counts[expert] = (masks[slot] >> lane) & 1u;
    }
  }
  if (picked) {
    // The route is the pick; the active list orders experts by index, so the rank counts the picked experts below.
    uint rank = 0;
    METAL_PRAGMA_UNROLL
    for (uint word = 0; word < ROUTE_EXPERTS_PER_LANE; ++word) {
      const uint below = word < picks.expert / METAL_SIMD_SIZE    ? 0xFFFFFFFFu
                         : word == picks.expert / METAL_SIMD_SIZE ? (1u << (picks.expert % METAL_SIMD_SIZE)) - 1u
                                                                  : 0u;
      rank += popcount(masks[word] & below);
    }
    tables.route_experts[lane] = picks.expert;
    tables.route_weights[lane] = picks.weight;
    tables.expert_routes[picks.expert] = lane;
    tables.active_experts[1 + rank] = active_entry(picks.expert, 1, lane);
  }
  if (lane == 0) {
    tables.active_experts[0] = params.top_k;
  }
}

// Selects every token's experts from its router logits and groups the routes by expert. One threadgroup runs it,
// with `simdgroups` simdgroups; `logit_at(index)` reads logits[index] of the [tokens, experts] matrix.
// `counts` and `first_routes` hold one slot per expert and `ranks` one slot per simdgroup.
template <typename LogitLoader>
METAL_FUNC void route_finalize(
    LogitLoader logit_at,
    const device float* expert_biases,
    const RouteParams params,
    const RouteTables tables,
    threadgroup atomic_uint* counts,
    threadgroup uint* first_routes,
    threadgroup uint* ranks,
    const uint thread_index,
    const uint threads,
    const uint simdgroup,
    const uint simdgroups,
    const uint lane
) {
  if (params.tokens == 1) {
    if (simdgroup == 0) {
      route_single_token(logit_at, expert_biases, params, tables, lane);
    }
    return;
  }
  for (uint expert = thread_index; expert < params.experts; expert += threads) {
    atomic_store_explicit(&counts[expert], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint token = simdgroup; token < params.tokens; token += simdgroups) {
    const TokenPicks picks = select_token_experts(logit_at, expert_biases, params, token, lane);
    if (lane < params.top_k) {
      const uint route = token * params.top_k + lane;
      tables.route_experts[route] = picks.expert;
      tables.route_weights[route] = picks.weight;
      const uint position = atomic_fetch_add_explicit(&counts[picks.expert], 1u, memory_order_relaxed);
      tables.expert_routes[picks.expert * params.tokens + position] = route;
      if (position == 0) {
        first_routes[picks.expert] = route;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Compact the experts that received a route, in increasing order.
  const uint rounds = (params.experts + threads - 1) / threads;
  uint base = 0;
  for (uint round = 0; round < rounds; ++round) {
    const uint expert = round * threads + thread_index;
    const uint count = expert < params.experts ? atomic_load_explicit(&counts[expert], memory_order_relaxed) : 0u;
    if (expert < params.experts) {
      tables.expert_counts[expert] = count;
    }
    const uint active = count > 0 ? 1u : 0u;
    const uint rank_in_simdgroup = simd_prefix_exclusive_sum(active);
    const uint simdgroup_total = simd_sum(active);
    if (lane == 0) {
      ranks[simdgroup] = simdgroup_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint rank = base + rank_in_simdgroup;
    uint round_total = 0;
    for (uint group = 0; group < simdgroups; ++group) {
      const uint group_total = ranks[group];
      rank += group < simdgroup ? group_total : 0u;
      round_total += group_total;
    }
    if (active != 0) {
      tables.active_experts[1 + rank] = active_entry(expert, count, first_routes[expert]);
    }
    base += round_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (thread_index == 0) {
    tables.active_experts[0] = base;
  }
}

} // namespace moe
} // namespace uzu
