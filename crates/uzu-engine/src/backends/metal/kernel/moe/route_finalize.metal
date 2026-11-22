#include <metal_stdlib>

#include "../common/defines.h"
#include "../common/dsl.h"
#include "../generated/moe.h"
#include "route_common.h"

using namespace metal;
using namespace uzu::moe;

#define FINALIZE_SIMDGROUPS 32
#define FINALIZE_MAX_EXPERTS 256

static_assert(FINALIZE_SIMDGROUPS * METAL_SIMD_SIZE == MOE_ROUTE_FINALIZE_THREADS, "one threadgroup routes the batch");
static_assert(FINALIZE_MAX_EXPERTS == MOE_MAX_EXPERTS, "the routing tables hold MOE_MAX_EXPERTS experts");

// Selects every token's experts from precomputed router logits and groups the routes by expert, in one
// threadgroup. Requires experts <= MOE_MAX_EXPERTS and 0 < top_k <= min(MOE_MAX_TOP_K, experts).
PUBLIC KERNEL(MoeRouteFinalize)(
    const device float* logits,
    const device float* expert_biases OPTIONAL(has_expert_biases),
    device uint* route_experts,
    device float* route_weights,
    device uint* expert_counts,
    device uint* expert_routes,
    device uint* active_experts,
    const constant uint& tokens,
    const constant uint& experts,
    const constant uint& top_k,
    const constant float& routed_scaling_factor,
    const bool has_expert_biases SPECIALIZE,
    const bool bias_selects SPECIALIZE,
    const bool bias_weights SPECIALIZE,
    threadgroup uint counts[FINALIZE_MAX_EXPERTS],
    threadgroup uint first_routes[FINALIZE_MAX_EXPERTS],
    threadgroup uint ranks[FINALIZE_SIMDGROUPS],
    const uint group GROUPS(1),
    const uint lane THREADS(METAL_SIMD_SIZE),
    const uint simdgroup THREADS(FINALIZE_SIMDGROUPS)
) {
  (void)group;
  const RouteParams params = {tokens, experts, top_k, routed_scaling_factor, bias_selects, bias_weights};
  const RouteTables tables = {route_experts, route_weights, expert_counts, expert_routes, active_experts};
  route_finalize(
      [&](uint index) { return logits[index]; },
      has_expert_biases ? expert_biases : nullptr,
      params,
      tables,
      reinterpret_cast<threadgroup atomic_uint*>(counts),
      first_routes,
      ranks,
      simdgroup * METAL_SIMD_SIZE + lane,
      FINALIZE_SIMDGROUPS * METAL_SIMD_SIZE,
      simdgroup,
      FINALIZE_SIMDGROUPS,
      lane
  );
}
