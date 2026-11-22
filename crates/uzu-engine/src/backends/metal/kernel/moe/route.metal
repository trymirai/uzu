#include <metal_stdlib>

#include "../common/defines.h"
#include "../common/dsl.h"
#include "../generated/moe.h"
#include "expert_common.h"
#include "route_common.h"

using namespace metal;
using namespace uzu::moe;

#define ROUTE_SIMDGROUPS 8
#define ROUTE_EXPERTS_PER_GROUP 4
#define ROUTE_MAX_EXPERTS 256
#define ROUTE_SIMDGROUPS_PER_EXPERT (ROUTE_SIMDGROUPS / ROUTE_EXPERTS_PER_GROUP)

static_assert(
    ROUTE_EXPERTS_PER_GROUP == MOE_ROUTE_EXPERTS_PER_GROUP,
    "the host sizes the grid from the shared constant"
);
static_assert(ROUTE_MAX_EXPERTS == MOE_MAX_EXPERTS, "the routing tables hold MOE_MAX_EXPERTS experts");

// Router of a small batch in one dispatch. Every threadgroup computes the float32 logits of
// ROUTE_EXPERTS_PER_GROUP experts for all tokens, reading its router rows once; the threadgroup that finishes last
// selects every token's experts and groups the routes by expert.
// Requires tokens <= MAX_TOKENS <= MOE_GEMV_MAX_TOKENS, experts % ROUTE_EXPERTS_PER_GROUP == 0,
// experts <= MOE_MAX_EXPERTS, 0 < top_k <= min(MOE_MAX_TOP_K, experts), model_dim % (8 * METAL_SIMD_SIZE) == 0,
// and finish_counter zero on entry. The last threadgroup leaves it zero again.
template <typename T, uint MAX_TOKENS>
VARIANTS(T, bfloat)
VARIANTS(MAX_TOKENS, 1, 2, 4, 8, 16, 32)
PUBLIC KERNEL(MoeRoute)(
    const device T* input,
    const device float* router_weights,
    const device float* router_biases OPTIONAL(has_router_biases),
    const device float* expert_biases OPTIONAL(has_expert_biases),
    device float* logits,
    device uint* route_experts,
    device float* route_weights,
    device uint* expert_counts,
    device uint* expert_routes,
    device uint* active_experts,
    device uint* finish_counter,
    const constant uint& tokens,
    const constant uint& model_dim,
    const constant uint& experts,
    const constant uint& top_k,
    const constant float& routed_scaling_factor,
    const bool has_router_biases SPECIALIZE,
    const bool has_expert_biases SPECIALIZE,
    const bool bias_selects SPECIALIZE,
    const bool bias_weights SPECIALIZE,
    threadgroup float partials[ROUTE_SIMDGROUPS * MAX_TOKENS],
    threadgroup uint counts[ROUTE_MAX_EXPERTS],
    threadgroup uint first_routes[ROUTE_MAX_EXPERTS],
    threadgroup uint ranks[ROUTE_SIMDGROUPS],
    threadgroup uint is_last[1],
    const uint group GROUPS(experts / ROUTE_EXPERTS_PER_GROUP),
    const uint lane THREADS(METAL_SIMD_SIZE),
    const uint simdgroup THREADS(ROUTE_SIMDGROUPS)
) {
  const uint expert_in_group = simdgroup / ROUTE_SIMDGROUPS_PER_EXPERT;
  const uint part = simdgroup % ROUTE_SIMDGROUPS_PER_EXPERT;
  const uint expert = group * ROUTE_EXPERTS_PER_GROUP + expert_in_group;
  const uint part_len = model_dim / ROUTE_SIMDGROUPS_PER_EXPERT;
  const uint vectors = part_len / 4;
  const device float4* weight_row =
      reinterpret_cast<const device float4*>(router_weights + size_t(expert) * model_dim + part * part_len);
  const device T* part_input = input + part * part_len;

  // Every router weight is read once, for all tokens.
  float acc[MAX_TOKENS];
  METAL_PRAGMA_UNROLL
  for (uint j = 0; j < MAX_TOKENS; ++j) {
    acc[j] = 0.0f;
  }
  const auto accumulate = [&](const uint vector) {
    const float4 w = weight_row[vector];
    METAL_PRAGMA_UNROLL
    for (uint j = 0; j < MAX_TOKENS; ++j) {
      // Slots past the batch repeat its last token, so that every load issues without a branch; their sums go
      // unused.
      const float4 x = load_float4(part_input + size_t(min(j, tokens - 1)) * model_dim + 4 * vector);
      acc[j] = fma(w.x, x.x, fma(w.y, x.y, fma(w.z, x.z, fma(w.w, x.w, acc[j]))));
    }
  };
  if (MAX_TOKENS == 1) {
#pragma clang loop unroll_count(9)
    for (uint vector = lane; vector < vectors; vector += METAL_SIMD_SIZE) {
      accumulate(vector);
    }
  } else {
#pragma clang loop unroll_count(3)
    for (uint vector = lane; vector < vectors; vector += METAL_SIMD_SIZE) {
      accumulate(vector);
    }
  }
  METAL_PRAGMA_UNROLL
  for (uint j = 0; j < MAX_TOKENS; ++j) {
    if (j < tokens) {
      const float total = simd_sum(acc[j]);
      if (lane == 0) {
        partials[simdgroup * MAX_TOKENS + j] = total;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint thread_index = simdgroup * METAL_SIMD_SIZE + lane;
  for (uint index = thread_index; index < ROUTE_EXPERTS_PER_GROUP * tokens;
       index += ROUTE_SIMDGROUPS * METAL_SIMD_SIZE) {
    const uint local_expert = index / tokens;
    const uint j = index % tokens;
    float logit = 0.0f;
    for (uint p = 0; p < ROUTE_SIMDGROUPS_PER_EXPERT; ++p) {
      logit += partials[(local_expert * ROUTE_SIMDGROUPS_PER_EXPERT + p) * MAX_TOKENS + j];
    }
    const uint out_expert = group * ROUTE_EXPERTS_PER_GROUP + local_expert;
    if (has_router_biases) {
      logit += router_biases[out_expert];
    }
    logits[j * experts + out_expert] = logit;
  }

  // Publish this group's logits, then count it; the group that brings the count to the total routes the batch.
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device);
  device atomic_uint* counter = reinterpret_cast<device atomic_uint*>(finish_counter);
  if (thread_index == 0) {
    const uint groups = experts / ROUTE_EXPERTS_PER_GROUP;
    const uint finished = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    is_last[0] = finished + 1 == groups ? 1u : 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (is_last[0] == 0) {
    return;
  }
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);

  const device atomic_uint* published = reinterpret_cast<const device atomic_uint*>(logits);
  const RouteParams params = {tokens, experts, top_k, routed_scaling_factor, bias_selects, bias_weights};
  const RouteTables tables = {route_experts, route_weights, expert_counts, expert_routes, active_experts};
  route_finalize(
      [&](uint index) { return as_type<float>(atomic_load_explicit(&published[index], memory_order_relaxed)); },
      has_expert_biases ? expert_biases : nullptr,
      params,
      tables,
      reinterpret_cast<threadgroup atomic_uint*>(counts),
      first_routes,
      ranks,
      thread_index,
      ROUTE_SIMDGROUPS * METAL_SIMD_SIZE,
      simdgroup,
      ROUTE_SIMDGROUPS,
      lane
  );
  if (thread_index == 0) {
    atomic_store_explicit(counter, 0u, memory_order_relaxed);
  }
}
