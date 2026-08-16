#include <metal_stdlib>
#include <metal_atomic>

#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"

template <ushort BLOCK_SIZE, typename T>
static T threadgroup_raking_prefix_exclusive_sum(T value, threadgroup T* shared, const ushort lid) {
  shared[lid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lid < 32) {
    const short values_per_thread = BLOCK_SIZE / 32;
    const short first_index = lid * values_per_thread;
    for (short i = first_index + 1; i < first_index + values_per_thread; i++) {
      shared[i] += shared[i - 1];
    }
    T partial_sum = shared[first_index + values_per_thread - 1];
    for (short i = first_index + values_per_thread - 1; i > first_index; i--) {
      shared[i] = shared[i - 1];
    }
    shared[first_index] = 0;

    T prefix = simd_prefix_exclusive_sum(partial_sum);

    for (short i = first_index; i < first_index + values_per_thread; i++) {
      shared[i] += prefix;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const T result = shared[lid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}

#define THREADGROUP_SIZE 128
#define SCATTER_BLOCK_SIZE 256
#define EXPERT_TILE_SIZE 512

// Single-kernel fused: count all experts + scan to offsets
// This kernel is launched with SINGLE threadgroup
PUBLIC KERNEL(MoeCountsOffsetsFused)(
    device const int* selected_expert_ids,
    device uint* expert_offsets,              // output: exclusive scan [expert_count + 1]
    device uint* total_routed_row_count,      // output: total count [1]
    device uint* scatter_block_expert_counts, // output: partial histograms for block_bases
    constant uint& token_count,
    constant uint& expert_count,
    constant uint& selected_experts_per_token,
    threadgroup _atomic<uint> threadgroup_expert_histogram[EXPERT_TILE_SIZE],
    threadgroup uint prefix_scan_scratch[THREADGROUP_SIZE],
    threadgroup uint reduction_scratch[THREADGROUP_SIZE],
    threadgroup uint total_expert_counts[EXPERT_TILE_SIZE], // Cache global counts in threadgroup memory
    threadgroup uint& expert_offset_carry,
    const ThreadContext thread_context,
    const uint thread_index_in_threadgroup THREADS(128)
) {
  if (expert_count == 0) {
    if (thread_index_in_threadgroup == 0) {
      expert_offsets[0] = 0u;
      total_routed_row_count[0] = 0u;
    }
    return;
  }

  // ═══════════════════════════════════════════════════════════
  // PHASE 1: Count tokens per expert using tiled histogram
  // ═══════════════════════════════════════════════════════════
  const uint scatter_block_count = (token_count + SCATTER_BLOCK_SIZE - 1) / SCATTER_BLOCK_SIZE;
  const uint expert_tile_count = (expert_count + EXPERT_TILE_SIZE - 1) / EXPERT_TILE_SIZE;

  // Tile over the expert dimension.
  for (uint expert_tile_start = 0; expert_tile_start < expert_count; expert_tile_start += EXPERT_TILE_SIZE) {
    const uint expert_tile_size =
        (expert_tile_start + EXPERT_TILE_SIZE <= expert_count) ? EXPERT_TILE_SIZE : (expert_count - expert_tile_start);
    const uint expert_tile_index = expert_tile_start / EXPERT_TILE_SIZE;

    for (uint expert_index_in_tile = thread_index_in_threadgroup; expert_index_in_tile < expert_tile_size;
         expert_index_in_tile += THREADGROUP_SIZE) {
      total_expert_counts[expert_tile_start + expert_index_in_tile] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scatter consumes one partial histogram per 256-token block.
    for (uint scatter_block_index = 0; scatter_block_index < scatter_block_count; ++scatter_block_index) {
      for (uint expert_index_in_tile = thread_index_in_threadgroup; expert_index_in_tile < expert_tile_size;
           expert_index_in_tile += THREADGROUP_SIZE) {
        atomic_store_explicit(&threadgroup_expert_histogram[expert_index_in_tile], 0u, memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const uint token_block_start = scatter_block_index * SCATTER_BLOCK_SIZE;
      const uint token_block_end = min(token_block_start + SCATTER_BLOCK_SIZE, token_count);
      for (uint token_index = token_block_start + thread_index_in_threadgroup; token_index < token_block_end;
           token_index += THREADGROUP_SIZE) {
        const uint selected_expert_row_start = token_index * selected_experts_per_token;
        for (uint selected_expert_slot = 0; selected_expert_slot < selected_experts_per_token; ++selected_expert_slot) {
          int selected_expert_id = selected_expert_ids[selected_expert_row_start + selected_expert_slot];
          if (selected_expert_id >= 0) {
            uint expert_index = uint(selected_expert_id);
            if (expert_index >= expert_tile_start && expert_index < expert_tile_start + expert_tile_size) {
              uint expert_index_in_tile = expert_index - expert_tile_start;
              atomic_fetch_add_explicit(&threadgroup_expert_histogram[expert_index_in_tile], 1u, memory_order_relaxed);
            }
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (uint expert_index_in_tile = thread_index_in_threadgroup; expert_index_in_tile < expert_tile_size;
           expert_index_in_tile += THREADGROUP_SIZE) {
        const uint scatter_block_expert_count =
            atomic_load_explicit(&threadgroup_expert_histogram[expert_index_in_tile], memory_order_relaxed);
        total_expert_counts[expert_tile_start + expert_index_in_tile] += scatter_block_expert_count;
        const uint partial_count_index =
            (scatter_block_index * expert_tile_count + expert_tile_index) * EXPERT_TILE_SIZE + expert_index_in_tile;
        scatter_block_expert_counts[partial_count_index] = scatter_block_expert_count;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // ═══════════════════════════════════════════════════════════
  // PHASE 2: Compute exclusive prefix scan on counts
  // ═══════════════════════════════════════════════════════════
  if (thread_index_in_threadgroup == 0) {
    expert_offset_carry = 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint expert_chunk_start = 0; expert_chunk_start < expert_count; expert_chunk_start += THREADGROUP_SIZE) {
    uint remaining_expert_count = expert_count - expert_chunk_start;
    uint expert_chunk_size = remaining_expert_count < THREADGROUP_SIZE ? remaining_expert_count : THREADGROUP_SIZE;

    uint expert_route_count = (thread_index_in_threadgroup < expert_chunk_size)
                                  ? total_expert_counts[expert_chunk_start + thread_index_in_threadgroup]
                                  : 0u;

    uint expert_offset_in_chunk = threadgroup_raking_prefix_exclusive_sum<THREADGROUP_SIZE>(
        expert_route_count,
        prefix_scan_scratch,
        (ushort)thread_index_in_threadgroup
    );
    uint expert_offset = expert_offset_in_chunk + expert_offset_carry;
    if (thread_index_in_threadgroup < expert_chunk_size) {
      expert_offsets[expert_chunk_start + thread_index_in_threadgroup] = expert_offset;
    }

    uint expert_chunk_route_count = threadgroup_cooperative_reduce<SimdReduceSum<uint>, THREADGROUP_SIZE>(
        expert_route_count,
        reduction_scratch,
        thread_context
    );
    if (thread_index_in_threadgroup == 0) {
      expert_offset_carry += expert_chunk_route_count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write final offset and total
  if (thread_index_in_threadgroup == 0) {
    expert_offsets[expert_count] = expert_offset_carry;
    total_routed_row_count[0] = expert_offset_carry;
  }
}
