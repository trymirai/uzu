#include <metal_stdlib>
#include <metal_atomic>
#include "../definitions.metal"
using namespace metal;

#define BLOCK_SIZE 256
#define TILE_E 512
#define SIMD_WIDTH 32
#define NUM_SG (BLOCK_SIZE / SIMD_WIDTH)

// 4B: Scatter into expert-major buckets using atomic-free device writes
template <typename ProbT>
inline void moe_scatter_buckets_impl(
    device const int* topk_ids,
    device const ProbT* topk_probs,
    device const uint* offsets,
    device const uint* block_bases,
    device const uint* block_alloc,
    device int* out_ids,
    device ProbT* out_probs,
    device int* tok2row_opt,
    uint T,
    uint E,
    uint K,
    uint num_blocks,
    uint num_tiles,
    uint3 tid3,
    uint lid,
    uint3 tgpig,
    threadgroup _atomic<uint>* sg_counts,
    threadgroup uint* sg_base
) {
  if (E == 0 || T == 0 || K == 0)
    return;
  const uint block_id = tgpig.x;
  const uint t_start = block_id * BLOCK_SIZE;
  const uint t_end = min(t_start + BLOCK_SIZE, T);
  const uint sg_id = lid / SIMD_WIDTH;
  const uint lane = lid % SIMD_WIDTH;

  // Per-tile processing
  for (uint e0 = 0; e0 < E; e0 += TILE_E) {
    const uint tile_e = (e0 + TILE_E <= E) ? TILE_E : (E - e0);

    // Zero counts and bases for active tile
    for (uint te = lid; te < tile_e; te += BLOCK_SIZE) {
      for (uint sg = 0; sg < NUM_SG; ++sg) {
        atomic_store_explicit(
            &sg_counts[sg * TILE_E + te],
            0u,
            memory_order_relaxed
        );
        sg_base[sg * TILE_E + te] = 0u;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 1: per-sg counting using threadgroup atomics (order-independent)
    for (uint t = t_start + lid; t < t_end; t += BLOCK_SIZE) {
      const uint base = t * K;
      for (uint k = 0; k < K; ++k) {
        int eid = topk_ids[base + k];
        if (eid >= 0) {
          const uint ue = (uint)eid;
          if (ue >= e0 && ue < e0 + tile_e) {
            const uint te = ue - e0;
            atomic_fetch_add_explicit(
                &sg_counts[sg_id * TILE_E + te],
                1u,
                memory_order_relaxed
            );
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: compute per-sg bases via prefix across sgs with capacity clamp
    for (uint te = lid; te < tile_e; te += BLOCK_SIZE) {
      const uint tile_id = e0 / TILE_E;
      const uint base_idx_block =
          (block_id * num_tiles + tile_id) * TILE_E + te;
      const uint alloc_total = block_alloc[base_idx_block];
      uint prefix = 0u;
      for (uint sg = 0; sg < NUM_SG; ++sg) {
        const uint idx = sg * TILE_E + te;
        const uint c =
            atomic_load_explicit(&sg_counts[idx], memory_order_relaxed);
        const uint room = (prefix < alloc_total) ? (alloc_total - prefix) : 0u;
        const uint take = min(c, room);
        sg_base[idx] = prefix;
        prefix += take;
      }
      // Reuse sg_counts as write indices: reset to zero
      for (uint sg = 0; sg < NUM_SG; ++sg) {
        atomic_store_explicit(
            &sg_counts[sg * TILE_E + te],
            0u,
            memory_order_relaxed
        );
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: deterministic write with 32-lane sequencing, no device atomics
    for (uint step = 0; step < SIMD_WIDTH; ++step) {
      if (lane == step) {
        const uint t = t_start + lid;
        if (t < t_end) {
          const uint base = t * K;
          for (uint k = 0; k < K; ++k) {
            int eid = topk_ids[base + k];
            if (eid >= 0) {
              const uint ue = (uint)eid;
              if (ue >= e0 && ue < e0 + tile_e) {
                const uint te = ue - e0;
                const uint tile_id = e0 / TILE_E;
                const uint base_idx_block =
                    (block_id * num_tiles + tile_id) * TILE_E + te;
                const uint idx_sg_te = sg_id * TILE_E + te;
                const uint local = atomic_fetch_add_explicit(
                    &sg_counts[idx_sg_te],
                    1u,
                    memory_order_relaxed
                );
                const uint base_s = sg_base[idx_sg_te];

                const uint slot =
                    offsets[ue] + block_bases[base_idx_block] + base_s + local;
                out_ids[slot] = (int)t;
                out_probs[slot] = topk_probs[base + k];
                if (tok2row_opt != nullptr) {
                  tok2row_opt[base + k] = (int)slot;
                }
              }
            }
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// 4A: Compute per-block base offsets from partials
// partials layout: index = (block_id * num_tiles + tile_id) * TILE_E + te
KERNEL(MoeBlockBasesFromPartials)(
    device const uint* partials,
    device uint* block_bases,
    device uint* block_alloc,
    constant uint& e_input,
    constant uint& num_blocks,
    constant uint& num_tiles,
    constant uint& capacity_per_expert, // 0 => no clamp
    const uint gid AXIS(num_tiles * 512, 256)
) {
  const uint tile_id = gid / TILE_E;
  const uint te = gid % TILE_E;
  const uint e = tile_id * TILE_E + te;
  if (e >= e_input)
    return;

  uint running = 0u;
  const bool clamp = (capacity_per_expert > 0u);
  for (uint b = 0; b < num_blocks; ++b) {
    const uint idx = (b * num_tiles + tile_id) * TILE_E + te;
    const uint p = partials[idx];
    const uint alloc = clamp ? (running < capacity_per_expert
                                    ? min(p, capacity_per_expert - running)
                                    : 0u)
                             : p;
    block_bases[idx] = running;
    block_alloc[idx] = alloc;
    running += alloc;
    if (clamp && running >= capacity_per_expert) {
      // Remainder allocate zero and base stays at capacity
      for (uint bb = b + 1; bb < num_blocks; ++bb) {
        const uint idx2 = (bb * num_tiles + tile_id) * TILE_E + te;
        block_bases[idx2] = running;
        block_alloc[idx2] = 0u;
      }
      break;
    }
  }
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MoeScatterBuckets)(
    device const int* topk_ids,
    device const T* topk_probs,
    device const uint* offsets,
    device const uint* block_bases,
    device const uint* block_alloc,
    device int* out_ids,
    device T* out_probs,
    constant uint& t,
    constant uint& e,
    constant uint& k,
    constant uint& num_blocks,
    constant uint& num_tiles,
    threadgroup _atomic<uint> sg_counts[NUM_SG * TILE_E],
    threadgroup uint sg_base[NUM_SG * TILE_E],
    const uint tgpig_x GROUPS(num_blocks),
    const uint lid THREADS(256)
) {
  const uint3 tgpig = {tgpig_x, 0, 0};
  const uint tid_x = tgpig.x * BLOCK_SIZE + lid;
  const uint3 tid3 = {tid_x, 0, 0};
  moe_scatter_buckets_impl<T>(
      topk_ids,
      topk_probs,
      offsets,
      block_bases,
      block_alloc,
      out_ids,
      out_probs,
      (device int*)nullptr,
      t,
      e,
      k,
      num_blocks,
      num_tiles,
      tid3,
      lid,
      tgpig,
      sg_counts,
      sg_base
  );
}

// Map variants
template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MoeScatterBucketsMap)(
    device const int* topk_ids,
    device const T* topk_probs,
    device const uint* offsets,
    device const uint* block_bases,
    device const uint* block_alloc,
    device int* out_ids,
    device T* out_probs,
    constant uint& t,
    constant uint& e,
    constant uint& k,
    constant uint& num_blocks,
    constant uint& num_tiles,
    device int* tok2row,
    threadgroup _atomic<uint> sg_counts[NUM_SG * TILE_E],
    threadgroup uint sg_base[NUM_SG * TILE_E],
    const uint tgpig_x GROUPS(num_blocks),
    const uint lid THREADS(256)
) {
  const uint3 tgpig = {tgpig_x, 0, 0};
  const uint tid_x = tgpig.x * BLOCK_SIZE + lid;
  const uint3 tid3 = {tid_x, 0, 0};
  moe_scatter_buckets_impl<T>(
      topk_ids,
      topk_probs,
      offsets,
      block_bases,
      block_alloc,
      out_ids,
      out_probs,
      tok2row,
      t,
      e,
      k,
      num_blocks,
      num_tiles,
      tid3,
      lid,
      tgpig,
      sg_counts,
      sg_base
  );
}