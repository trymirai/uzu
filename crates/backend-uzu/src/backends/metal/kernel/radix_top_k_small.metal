#include <metal_stdlib>
#include <metal_atomic>
#include "common/defines.h"
#include "common/dsl.h"
#include "common/top_k.h"

using namespace metal;

constant uint THREADS_PER_TG = 256;
constant uint MAX_K = 512;
constant uint RADIX_BITS = 10;
constant uint BUCKETS = 1 << RADIX_BITS;
constant uint BUCKETS_PER_THREAD = BUCKETS / THREADS_PER_TG;

static bool arrive_last(device atomic_uint* count, uint expected) {
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  const bool last = atomic_fetch_add_explicit(count, 1u, memory_order_relaxed) == expected - 1;
  if (last)
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  return last;
}

static void reset_arrival(device atomic_uint* count) {
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  atomic_store_explicit(count, 0u, memory_order_relaxed);
}

template <typename Visitor>
static inline void visit_partition_keys(
    const device float* input,
    uint columns,
    uint index_bits,
    uint partition,
    uint partitions,
    uint lid,
    Visitor visit
) {
  if (columns % 4u != 0) {
    const uint begin = columns * partition / partitions;
    const uint end = columns * (partition + 1u) / partitions;
    for (uint column = begin + lid; column < end; column += THREADS_PER_TG)
      visit(top_k_ordered_key(input[column], column, index_bits));
    return;
  }
  const device float4* input4 = reinterpret_cast<const device float4*>(input);
  const uint vector_columns = columns / 4u;
  const uint vector_begin = vector_columns * partition / partitions;
  const uint vector_end = vector_columns * (partition + 1u) / partitions;
  for (uint vector = vector_begin + lid; vector < vector_end; vector += THREADS_PER_TG) {
    const float4 values = input4[vector];
    for (uint lane = 0; lane < 4; ++lane)
      visit(top_k_ordered_key(values[lane], vector * 4 + lane, index_bits));
  }
  if (partition + 1 == partitions) {
    for (uint column = vector_columns * 4 + lid; column < columns; column += THREADS_PER_TG)
      visit(top_k_ordered_key(input[column], column, index_bits));
  }
}

KERNEL(RadixTopKSmallPass)(
    const device float* input,
    device atomic_uint* partial_histograms,
    device ulong* prefixes,
    device ulong* prefix_masks,
    device uint* ranks,
    device atomic_uint* done_counts,
    constant uint& rows,
    constant uint& k,
    constant uint& pass,
    const uint columns SPECIALIZE,
    const uint partitions SPECIALIZE,
    threadgroup uint histogram[BUCKETS],
    threadgroup uint simd_prefixes[THREADS_PER_TG / METAL_SIMD_SIZE],
    threadgroup uint& is_last,
    const uint group GROUPS(rows* partitions),
    const uint lid THREADS(256)
) {
  const uint row = group / partitions;
  const uint partition = group % partitions;
  const uint index_bits = columns <= 1 ? 1u : 32u - clz(columns - 1u);
  const uint passes = (32u + index_bits + RADIX_BITS - 1u) / RADIX_BITS;
  const uint shift = (passes - pass - 1u) * RADIX_BITS;
  const device float* row_input = input + ulong(row) * columns;
  const ulong prefix = prefixes[row];
  const ulong mask = prefix_masks[row];

  // Build this partition's histogram.
  device atomic_uint* partition_histogram = partial_histograms + (row * partitions + partition) * BUCKETS;
  for (uint bucket = lid; bucket < BUCKETS; bucket += THREADS_PER_TG)
    atomic_store_explicit(&partition_histogram[bucket], 0u, memory_order_relaxed);
  // Order this partition's clears before its histogram updates.
  threadgroup_barrier(mem_flags::mem_device);
  visit_partition_keys(row_input, columns, index_bits, partition, partitions, lid, [&](ulong key) {
    if ((key & mask) == prefix)
      atomic_fetch_add_explicit(&partition_histogram[(key >> shift) & (BUCKETS - 1)], 1u, memory_order_relaxed);
  });
  threadgroup_barrier(mem_flags::mem_device);

  // The last partition merges the row.
  if (lid == 0)
    is_last = arrive_last(&done_counts[row], partitions);
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  if (!is_last)
    return;

  for (uint bucket = lid; bucket < BUCKETS; bucket += THREADS_PER_TG) {
    uint count = 0;
    for (uint p = 0; p < partitions; ++p)
      count +=
          atomic_load_explicit(&partial_histograms[(row * partitions + p) * BUCKETS + bucket], memory_order_relaxed);
    histogram[bucket] = count;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint sum = 0;
  const uint begin = BUCKETS - (lid + 1) * BUCKETS_PER_THREAD;
  for (uint offset = 0; offset < BUCKETS_PER_THREAD; ++offset)
    sum += histogram[begin + offset];
  uint higher = simd_prefix_exclusive_sum(sum);
  if ((lid + 1) % METAL_SIMD_SIZE == 0)
    simd_prefixes[lid / METAL_SIMD_SIZE] = higher + sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < METAL_SIMD_SIZE) {
    const uint simdgroups = THREADS_PER_TG / METAL_SIMD_SIZE;
    const uint simd_sum = lid < simdgroups ? simd_prefixes[lid] : 0;
    const uint simd_prefix = simd_prefix_exclusive_sum(simd_sum);
    if (lid < simdgroups)
      simd_prefixes[lid] = simd_prefix;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  higher += simd_prefixes[lid / METAL_SIMD_SIZE];

  uint rank = pass == 0 ? k - 1 : ranks[row];
  if (higher <= rank && rank < higher + sum) {
    rank -= higher;
    const uint group_begin = BUCKETS - (lid + 1) * BUCKETS_PER_THREAD;
    for (uint offset = 0; offset < BUCKETS_PER_THREAD; ++offset) {
      const uint bucket = group_begin + BUCKETS_PER_THREAD - offset - 1;
      const uint count = histogram[bucket];
      if (rank < count) {
        prefixes[row] |= ulong(bucket) << shift;
        prefix_masks[row] |= ulong(BUCKETS - 1) << shift;
        break;
      }
      rank -= count;
    }
    ranks[row] = rank;
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (lid == 0)
    reset_arrival(&done_counts[row]);
}

KERNEL(RadixTopKSmallCollect)(
    const device float* input,
    device uint* output_ids,
    device float* output_scores,
    const device ulong* prefixes,
    device atomic_uint* selected_counts,
    device atomic_uint* done_counts,
    device ulong* selected_keys,
    constant uint& rows,
    constant uint& k,
    const uint columns SPECIALIZE,
    const uint partitions SPECIALIZE,
    threadgroup ulong sorted_keys[MAX_K],
    threadgroup uint& is_last,
    const uint group GROUPS(rows* partitions),
    const uint lid THREADS(256)
) {
  const uint row = group / partitions;
  const uint partition = group % partitions;
  const uint index_bits = columns <= 1 ? 1u : 32u - clz(columns - 1u);
  const device float* row_input = input + ulong(row) * columns;
  const ulong prefix = prefixes[row];

  visit_partition_keys(row_input, columns, index_bits, partition, partitions, lid, [&](ulong key) {
    if (key >= prefix) {
      const uint index = atomic_fetch_add_explicit(&selected_counts[row], 1u, memory_order_relaxed);
      // The final radix prefix selects exactly k unique keys.
      selected_keys[row * k + index] = key;
    }
  });
  threadgroup_barrier(mem_flags::mem_device);
  if (lid == 0)
    is_last = arrive_last(&done_counts[row], partitions);
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  if (!is_last)
    return;

  for (uint index = lid; index < MAX_K; index += THREADS_PER_TG)
    sorted_keys[index] = index < k ? selected_keys[row * k + index] : 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint size = 2; size <= MAX_K; size <<= 1) {
    for (uint stride = size >> 1; stride; stride >>= 1) {
      for (uint index = lid; index < MAX_K; index += THREADS_PER_TG) {
        const uint other = index ^ stride;
        if (other > index) {
          const bool descending = (index & size) == 0;
          if (descending ? sorted_keys[index] < sorted_keys[other] : sorted_keys[index] > sorted_keys[other]) {
            const ulong key = sorted_keys[index];
            sorted_keys[index] = sorted_keys[other];
            sorted_keys[other] = key;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  for (uint index = lid; index < k; index += THREADS_PER_TG) {
    const ulong key = sorted_keys[index];
    const ulong mask = (1ul << index_bits) - 1;
    output_ids[row * k + index] = uint(mask - (key & mask));
    output_scores[row * k + index] = top_k_score_from_key(uint(key >> index_bits));
  }
}
