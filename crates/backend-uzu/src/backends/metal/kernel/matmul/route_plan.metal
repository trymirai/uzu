#include <metal_atomic>
#include <metal_stdlib>

#include "../common/dsl.h"

using namespace metal;

KERNEL(ExpertRouteClearCounts)(
    device _atomic<uint>* counts,
    constant uint& expert_count,
    const uint slot AXIS(expert_count + 1, 256)
) {
  atomic_store_explicit(&counts[slot], 0u, memory_order_relaxed);
}

KERNEL(ExpertRouteCount)(
    device const int* expert_ids,
    device _atomic<uint>* counts,
    constant uint& route_count,
    constant uint& expert_count,
    const uint route AXIS(route_count, 256)
) {
  const int expert = expert_ids[route];
  if (expert < 0 || uint(expert) >= expert_count) {
    return;
  }
  // Slot zero stays empty so the prefix pass can transform counts in place.
  atomic_fetch_add_explicit(&counts[uint(expert) + 1], 1u, memory_order_relaxed);
}

KERNEL(ExpertRoutePrefix)(
    device _atomic<uint>* counts_and_offsets,
    device _atomic<uint>* cursors,
    constant uint& expert_count,
    const uint lid THREADS(1),
    const uint group GROUPS(1)
) {
  (void)lid;
  (void)group;
  // One thread is intentional: expert banks are small relative to their
  // matrices, and this establishes deterministic contiguous segments.
  uint prefix = 0;
  for (uint expert = 0; expert < expert_count; ++expert) {
    const uint count = atomic_load_explicit(&counts_and_offsets[expert + 1], memory_order_relaxed);
    atomic_store_explicit(&counts_and_offsets[expert], prefix, memory_order_relaxed);
    atomic_store_explicit(&cursors[expert], prefix, memory_order_relaxed);
    prefix += count;
  }
  atomic_store_explicit(&counts_and_offsets[expert_count], prefix, memory_order_relaxed);
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

template <typename DT>
VARIANTS(DT, half, bfloat, float)
KERNEL(ExpertRouteZeroInvalid)(
    device const int* expert_ids,
    device DT* output,
    constant uint& route_count,
    constant uint& output_width,
    constant uint& expert_count,
    const uint route GROUPS(route_count),
    const uint lane THREADS(32)
) {
  const int expert = expert_ids[route];
  if (expert >= 0 && uint(expert) < expert_count) {
    return;
  }
  for (uint column = lane; column < output_width; column += 32) {
    output[ulong(route) * output_width + column] = DT(0);
  }
}
