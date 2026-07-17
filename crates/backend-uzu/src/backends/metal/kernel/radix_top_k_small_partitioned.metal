#include <metal_stdlib>
#include <metal_atomic>
#include "common/dsl.h"
#include "common/top_k.h"

using namespace metal;

constant uint THREADS = 256;
constant uint MAX_K = 512;
constant uint RADIX_BITS = 10;
constant uint BUCKETS = 1 << RADIX_BITS;
constant uint BUCKETS_PER_THREAD = BUCKETS / THREADS;

KERNEL(RadixTopKSmallPartitioned)(
    const device float* input,
    device uint* output_ids,
    device float* output_scores,
    device atomic_uint* partial_histograms,
    device ulong* prefixes,
    device ulong* prefix_masks,
    device uint* ranks,
    device atomic_uint* selected_counts,
    device ulong* selected_keys,
    constant uint& rows,
    constant uint& k,
    constant uint& pass,
    const uint columns SPECIALIZE,
    const uint partitions SPECIALIZE,
    const uint phase SPECIALIZE,
    threadgroup uint histogram[BUCKETS],
    threadgroup uint group_sums[THREADS],
    threadgroup ulong sorted_keys[MAX_K],
    const uint group GROUPS(rows* partitions),
    const uint lid THREADS(256)
) {
  const uint row = group / partitions;
  const uint partition = group % partitions;
  const uint index_bits = columns <= 1 ? 1u : 32u - clz(columns - 1u);
  const device float* row_input = input + ulong(row) * columns;
  const device float4* row_input4 = reinterpret_cast<const device float4*>(row_input);
  const uint vector_columns = columns / 4u;
  const uint vector_begin = vector_columns * partition / partitions;
  const uint vector_end = vector_columns * (partition + 1u) / partitions;

  if (phase == 0) {
    device atomic_uint* partial = partial_histograms + (row * partitions + partition) * BUCKETS;
    for (uint bucket = lid; bucket < BUCKETS; bucket += THREADS) {
      atomic_store_explicit(&partial[bucket], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);
    const uint passes = (32u + index_bits + RADIX_BITS - 1u) / RADIX_BITS;
    const uint shift = (passes - pass - 1u) * RADIX_BITS;
    const ulong prefix = prefixes[row];
    const ulong mask = prefix_masks[row];
    for (uint vector = vector_begin + lid; vector < vector_end; vector += THREADS) {
      const float4 values = row_input4[vector];
      for (uint lane = 0; lane < 4; ++lane) {
        const uint column = vector * 4 + lane;
        const ulong key = top_k_ordered_key(values[lane], column, index_bits);
        if ((key & mask) == prefix) {
          atomic_fetch_add_explicit(&partial[(key >> shift) & (BUCKETS - 1)], 1u, memory_order_relaxed);
        }
      }
    }
    if (partition + 1 == partitions) {
      for (uint column = vector_columns * 4 + lid; column < columns; column += THREADS) {
        const ulong key = top_k_ordered_key(row_input[column], column, index_bits);
        if ((key & mask) == prefix) {
          atomic_fetch_add_explicit(&partial[(key >> shift) & (BUCKETS - 1)], 1u, memory_order_relaxed);
        }
      }
    }
    return;
  }

  if (phase == 1) {
    if (partition != 0)
      return;
    for (uint bucket = lid; bucket < BUCKETS; bucket += THREADS) {
      uint count = 0;
      for (uint p = 0; p < partitions; ++p) {
        count +=
            atomic_load_explicit(&partial_histograms[(row * partitions + p) * BUCKETS + bucket], memory_order_relaxed);
      }
      histogram[bucket] = count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint sum = 0;
    const uint begin = BUCKETS - (lid + 1) * BUCKETS_PER_THREAD;
    for (uint offset = 0; offset < BUCKETS_PER_THREAD; ++offset)
      sum += histogram[begin + offset];
    group_sums[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
      uint rank = pass == 0 ? k - 1 : ranks[row];
      const uint passes = (32u + index_bits + RADIX_BITS - 1u) / RADIX_BITS;
      const uint shift = (passes - pass - 1u) * RADIX_BITS;
      for (uint g = 0; g < THREADS; ++g) {
        if (rank < group_sums[g]) {
          const uint group_begin = BUCKETS - (g + 1) * BUCKETS_PER_THREAD;
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
          break;
        }
        rank -= group_sums[g];
      }
      ranks[row] = rank;
    }
    return;
  }

  if (phase == 2) {
    const ulong prefix = prefixes[row];
    for (uint vector = vector_begin + lid; vector < vector_end; vector += THREADS) {
      const float4 values = row_input4[vector];
      for (uint lane = 0; lane < 4; ++lane) {
        const uint column = vector * 4 + lane;
        const ulong key = top_k_ordered_key(values[lane], column, index_bits);
        if (key >= prefix) {
          const uint index = atomic_fetch_add_explicit(&selected_counts[row], 1u, memory_order_relaxed);
          selected_keys[row * k + index] = key;
        }
      }
    }
    if (partition + 1 == partitions) {
      for (uint column = vector_columns * 4 + lid; column < columns; column += THREADS) {
        const ulong key = top_k_ordered_key(row_input[column], column, index_bits);
        if (key >= prefix) {
          const uint index = atomic_fetch_add_explicit(&selected_counts[row], 1u, memory_order_relaxed);
          selected_keys[row * k + index] = key;
        }
      }
    }
    return;
  }

  if (partition != 0)
    return;
  for (uint index = lid; index < MAX_K; index += THREADS)
    sorted_keys[index] = index < k ? selected_keys[row * k + index] : 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint size = 2; size <= MAX_K; size <<= 1) {
    for (uint stride = size >> 1; stride; stride >>= 1) {
      for (uint index = lid; index < MAX_K; index += THREADS) {
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
  for (uint index = lid; index < k; index += THREADS) {
    const ulong key = sorted_keys[index];
    const ulong mask = (1ul << index_bits) - 1;
    output_ids[row * k + index] = uint(mask - (key & mask));
    output_scores[row * k + index] = top_k_score_from_key(uint(key >> index_bits));
  }
}
