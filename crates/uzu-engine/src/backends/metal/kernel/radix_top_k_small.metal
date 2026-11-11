#include <metal_stdlib>
#include <metal_atomic>
#include "common/defines.h"
#include "common/dsl.h"
#include "common/thread_context.h"
#include "common/top_k.h"

using namespace metal;

#define THREADS_PER_TG 512
#define SIMDGROUPS_PER_TG (THREADS_PER_TG / METAL_SIMD_SIZE)

constant uint MAX_K = 512;
constant uint RADIX_BITS = 8;
constant uint BUCKETS = 1 << RADIX_BITS;
constant uint HISTOGRAM_REPLICAS = 8;
constant uint HISTOGRAM_SIZE = BUCKETS * HISTOGRAM_REPLICAS;
constant uint VECTOR_WIDTH = 8;
constant uint VECTORS_PER_THREAD = 2;
constant uint SCORES_PER_THREAD = VECTOR_WIDTH * VECTORS_PER_THREAD;
constant uint SCORES_PER_BLOCK = SCORES_PER_THREAD * THREADS_PER_TG;
constant uint HALF_BITS = 16;
constant uint HALF_MASK = (1 << HALF_BITS) - 1;

METAL_FUNC void load_row_keys(
    const device ushort* row_scores,
    uint columns,
    uint first_column,
    thread uint (&keys)[SCORES_PER_THREAD]
) {
  const bool vector_aligned = (reinterpret_cast<ulong>(row_scores) & 15) == 0;
  METAL_PRAGMA_UNROLL
  for (uint vector = 0; vector < VECTORS_PER_THREAD; ++vector) {
    const uint vector_column = first_column + vector * VECTOR_WIDTH;
    if (vector_aligned && vector_column + VECTOR_WIDTH <= columns) {
      const uint4 packed_scores = *reinterpret_cast<const device uint4*>(row_scores + vector_column);
      METAL_PRAGMA_UNROLL
      for (uint lane = 0; lane < VECTOR_WIDTH; ++lane) {
        keys[vector * VECTOR_WIDTH + lane] = top_k_bf16_key(ushort(packed_scores[lane / 2] >> (16 * (lane % 2))));
      }
    } else {
      for (uint lane = 0; lane < VECTOR_WIDTH; ++lane) {
        const uint column = vector_column + lane;
        keys[vector * VECTOR_WIDTH + lane] = column < columns ? top_k_bf16_key(row_scores[column]) : 0;
      }
    }
  }
}

METAL_FUNC uint2
threadgroup_prefix_sum(uint value, threadgroup uint* simdgroup_totals, const ThreadContext thread_context) {
  const uint simd_prefix = simd_prefix_exclusive_sum(value);
  if (thread_context.simd_lane_id == METAL_SIMD_SIZE - 1) {
    simdgroup_totals[thread_context.simdgroup_index] = simd_prefix + value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint previous_simdgroups_total = 0;
  uint total = 0;
  for (uint simdgroup = 0; simdgroup < SIMDGROUPS_PER_TG; ++simdgroup) {
    const uint simdgroup_total = simdgroup_totals[simdgroup];
    previous_simdgroups_total += simdgroup < thread_context.simdgroup_index ? simdgroup_total : 0;
    total += simdgroup_total;
  }
  return uint2(simd_prefix + previous_simdgroups_total, total);
}

KERNEL(RadixTopKSmall)(
    const device bfloat* input,
    device uint* output_ids,
    device float* output_scores,
    constant uint& rows,
    constant uint& columns,
    constant uint& k,
    const bool sorted SPECIALIZE,
    threadgroup atomic_uint histograms[2 * HISTOGRAM_SIZE],
    threadgroup uint simdgroup_totals[2 * SIMDGROUPS_PER_TG],
    threadgroup uint& chosen_bucket,
    threadgroup uint& chosen_rank,
    threadgroup uint winner_orders[MAX_K],
    threadgroup uint winner_columns[MAX_K],
    const ThreadContext thread_context,
    const uint row GROUPS(rows),
    const uint lid THREADS(THREADS_PER_TG)
) {
  const device bfloat* row_input = input + ulong(row) * columns;
  const device ushort* row_scores = reinterpret_cast<const device ushort*>(row_input);
  device uint* row_ids = output_ids + row * k;
  device float* row_output_scores = output_scores + row * k;

  for (uint index = lid; index < 2 * HISTOGRAM_SIZE; index += THREADS_PER_TG) {
    atomic_store_explicit(&histograms[index], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint threshold_key = 0;
  uint rank = k - 1;
  METAL_PRAGMA_UNROLL
  for (uint pass = 0; pass < 2; ++pass) {
    const uint shift = RADIX_BITS * (1 - pass);
    threadgroup atomic_uint* histogram = histograms + pass * HISTOGRAM_SIZE;
    for (uint first_column = lid * SCORES_PER_THREAD; first_column < columns; first_column += SCORES_PER_BLOCK) {
      uint keys[SCORES_PER_THREAD];
      load_row_keys(row_scores, columns, first_column, keys);
      METAL_PRAGMA_UNROLL
      for (uint index = 0; index < SCORES_PER_THREAD; ++index) {
        if ((keys[index] >> (shift + RADIX_BITS)) == threshold_key) {
          const uint replica = thread_context.simd_lane_id % HISTOGRAM_REPLICAS;
          const uint bucket = (keys[index] >> shift) & (BUCKETS - 1);
          atomic_fetch_add_explicit(&histogram[bucket * HISTOGRAM_REPLICAS + replica], 1u, memory_order_relaxed);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint bucket = BUCKETS - 1 - lid;
    uint count = 0;
    if (lid < BUCKETS) {
      for (uint replica = 0; replica < HISTOGRAM_REPLICAS; ++replica) {
        count += atomic_load_explicit(&histogram[bucket * HISTOGRAM_REPLICAS + replica], memory_order_relaxed);
      }
    }
    const uint larger_keys = threadgroup_prefix_sum(count, simdgroup_totals, thread_context).x;
    if (larger_keys <= rank && rank < larger_keys + count) {
      chosen_bucket = bucket;
      chosen_rank = rank - larger_keys;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threshold_key = threshold_key << RADIX_BITS | chosen_bucket;
    rank = chosen_rank;
  }

  const uint tie_winner_count = rank + 1;
  const uint above_threshold_count = k - tie_winner_count;
  uint above_threshold_before = 0;
  uint ties_before = 0;
  for (uint block = 0; block * SCORES_PER_BLOCK < columns; ++block) {
    const uint first_column = block * SCORES_PER_BLOCK + lid * SCORES_PER_THREAD;
    uint keys[SCORES_PER_THREAD];
    load_row_keys(row_scores, columns, first_column, keys);
    uint thread_above_threshold = 0;
    uint thread_ties = 0;
    METAL_PRAGMA_UNROLL
    for (uint index = 0; index < SCORES_PER_THREAD; ++index) {
      thread_above_threshold += keys[index] > threshold_key;
      thread_ties += keys[index] == threshold_key && first_column + index < columns;
    }
    const uint2 prefix_and_total = threadgroup_prefix_sum(
        thread_above_threshold | thread_ties << HALF_BITS,
        simdgroup_totals + (block % 2) * SIMDGROUPS_PER_TG,
        thread_context
    );
    uint above_threshold_slot = above_threshold_before + (prefix_and_total.x & HALF_MASK);
    uint tie_rank = ties_before + (prefix_and_total.x >> HALF_BITS);
    METAL_PRAGMA_UNROLL
    for (uint index = 0; index < SCORES_PER_THREAD; ++index) {
      const uint column = first_column + index;
      const bool above_threshold = keys[index] > threshold_key;
      const bool tie = keys[index] == threshold_key && column < columns;
      const uint slot = above_threshold_slot + min(tie_rank, tie_winner_count);
      if (above_threshold || (tie && tie_rank < tie_winner_count)) {
        if (sorted) {
          winner_orders[slot] = keys[index] << HALF_BITS | (HALF_MASK - slot);
          winner_columns[slot] = column;
        } else {
          row_ids[slot] = column;
          row_output_scores[slot] = float(row_input[column]);
        }
      }
      above_threshold_slot += above_threshold;
      tie_rank += tie;
    }
    above_threshold_before += prefix_and_total.y & HALF_MASK;
    ties_before += prefix_and_total.y >> HALF_BITS;
    if (above_threshold_before == above_threshold_count && ties_before >= tie_winner_count) {
      break;
    }
  }

  if (sorted) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint slot = lid; slot < k; slot += THREADS_PER_TG) {
      uint winners_ahead = 0;
      for (uint other_slot = 0; other_slot < k; ++other_slot) {
        winners_ahead += winner_orders[other_slot] > winner_orders[slot];
      }
      row_ids[winners_ahead] = winner_columns[slot];
      row_output_scores[winners_ahead] = float(row_input[winner_columns[slot]]);
    }
  }
}
