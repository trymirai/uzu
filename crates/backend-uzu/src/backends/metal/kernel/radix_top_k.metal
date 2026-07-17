#include <metal_stdlib>
#include <metal_atomic>
#include "common/dsl.h"
#include "common/top_k.h"

using namespace metal;

constant uint RADIX_TOP_K_THREADS = 256;
constant uint RADIX_TOP_K_MAX = 512;
constant uint RADIX_TOP_K_BITS = 10;
constant uint RADIX_TOP_K_SIZE = 1 << RADIX_TOP_K_BITS;

static inline ulong radix_top_k_ordered_key(float score, uint token, uint token_bits) {
  const uint score_key = top_k_score_key(score);
  const ulong token_mask = (1ul << token_bits) - 1ul;
  return (ulong(score_key) << token_bits) | (token_mask - ulong(token));
}

PUBLIC KERNEL(RadixTopK)(
    const device float* input,
    device uint* output_ids,
    device float* output_scores,
    constant uint& rows,
    constant uint& columns,
    constant uint& k,
    threadgroup _atomic<uint> histogram[RADIX_TOP_K_SIZE],
    threadgroup _atomic<uint>& selected_count,
    threadgroup ulong selected_keys[RADIX_TOP_K_MAX],
    threadgroup uint selected_ids[RADIX_TOP_K_MAX],
    threadgroup float selected_scores[RADIX_TOP_K_MAX],
    threadgroup ulong& prefix,
    threadgroup ulong& prefix_mask,
    threadgroup uint& rank,
    const uint row GROUPS(rows),
    const uint lid THREADS(256)
) {
  if (k == 0 || k > RADIX_TOP_K_MAX || k > columns) {
    return;
  }

  const device float* row_input = input + ulong(row) * ulong(columns);
  const uint token_bits = columns <= 1 ? 1u : 32u - clz(columns - 1u);
  const uint key_bits = 32u + token_bits;
  const uint radix_passes = (key_bits + RADIX_TOP_K_BITS - 1u) / RADIX_TOP_K_BITS;
  if (lid == 0) {
    prefix = 0;
    prefix_mask = 0;
    rank = k - 1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint pass = 0; pass < radix_passes; ++pass) {
    for (uint bucket = lid; bucket < RADIX_TOP_K_SIZE; bucket += RADIX_TOP_K_THREADS) {
      atomic_store_explicit(&histogram[bucket], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint shift = (radix_passes - pass - 1u) * RADIX_TOP_K_BITS;
    for (uint token = lid; token < columns; token += RADIX_TOP_K_THREADS) {
      const ulong key = radix_top_k_ordered_key(row_input[token], token, token_bits);
      if ((key & prefix_mask) == prefix) {
        const uint bucket = uint((key >> shift) & ulong(RADIX_TOP_K_SIZE - 1u));
        atomic_fetch_add_explicit(&histogram[bucket], 1u, memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
      for (int bucket = int(RADIX_TOP_K_SIZE) - 1; bucket >= 0; --bucket) {
        const uint count = atomic_load_explicit(&histogram[uint(bucket)], memory_order_relaxed);
        if (rank < count) {
          prefix |= ulong(uint(bucket)) << shift;
          prefix_mask |= ulong(RADIX_TOP_K_SIZE - 1u) << shift;
          break;
        }
        rank -= count;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint index = lid; index < RADIX_TOP_K_MAX; index += RADIX_TOP_K_THREADS) {
    selected_keys[index] = 0;
    selected_ids[index] = 0;
    selected_scores[index] = -INFINITY;
  }
  if (lid == 0) {
    atomic_store_explicit(&selected_count, 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint token = lid; token < columns; token += RADIX_TOP_K_THREADS) {
    const float score = row_input[token];
    const ulong key = radix_top_k_ordered_key(score, token, token_bits);
    if (key >= prefix) {
      const uint index = atomic_fetch_add_explicit(&selected_count, 1u, memory_order_relaxed);
      if (index < RADIX_TOP_K_MAX) {
        selected_keys[index] = key;
        selected_ids[index] = token;
        selected_scores[index] = score;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint size = 2; size <= RADIX_TOP_K_MAX; size <<= 1) {
    for (uint stride = size >> 1; stride > 0; stride >>= 1) {
      for (uint index = lid; index < RADIX_TOP_K_MAX; index += RADIX_TOP_K_THREADS) {
        const uint other = index ^ stride;
        if (other > index) {
          const bool descending = (index & size) == 0;
          const bool swap =
              descending ? selected_keys[index] < selected_keys[other] : selected_keys[index] > selected_keys[other];
          if (swap) {
            const ulong key = selected_keys[index];
            const uint token = selected_ids[index];
            const float score = selected_scores[index];
            selected_keys[index] = selected_keys[other];
            selected_ids[index] = selected_ids[other];
            selected_scores[index] = selected_scores[other];
            selected_keys[other] = key;
            selected_ids[other] = token;
            selected_scores[other] = score;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  for (uint index = lid; index < k; index += RADIX_TOP_K_THREADS) {
    output_ids[row * k + index] = selected_ids[index];
    output_scores[row * k + index] = selected_scores[index];
  }
}
