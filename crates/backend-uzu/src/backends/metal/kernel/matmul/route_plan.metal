#include <metal_atomic>
#include <metal_stdlib>

#include "../common/dsl.h"

using namespace metal;

#define ROUTED_MAX_EXPERTS 512

KERNEL(ExpertRouteCount)(
    device const int* expert_ids,
    device uint* offsets,
    device _atomic<uint>* cursors,
    constant uint& route_count,
    constant uint& expert_count,
    threadgroup _atomic<uint> histogram[ROUTED_MAX_EXPERTS],
    const uint lid THREADS(256),
    const uint group GROUPS(1)
) {
  (void)group;
  for (uint expert = lid; expert < expert_count; expert += 256) {
    atomic_store_explicit(&histogram[expert], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint route = lid; route < route_count; route += 256) {
    const int expert = expert_ids[route];
    if (expert >= 0 && uint(expert) < expert_count) {
      atomic_fetch_add_explicit(&histogram[uint(expert)], 1u, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lid == 0) {
    uint prefix = 0;
    for (uint expert = 0; expert < expert_count; ++expert) {
      offsets[expert] = prefix;
      atomic_store_explicit(&cursors[expert], prefix, memory_order_relaxed);
      prefix += atomic_load_explicit(&histogram[expert], memory_order_relaxed);
    }
    offsets[expert_count] = prefix;
  }
}

KERNEL(ExpertRouteScatter)(
    device const int* expert_ids,
    device _atomic<uint>* cursors,
    device uint* grouped_routes,
    constant uint& route_count,
    constant uint& expert_count,
    const uint route AXIS(route_count, 256)
) {
  const int expert = expert_ids[route];
  if (expert < 0 || uint(expert) >= expert_count) {
    return;
  }
  const uint destination = atomic_fetch_add_explicit(&cursors[uint(expert)], 1u, memory_order_relaxed);
  grouped_routes[destination] = route;
}
