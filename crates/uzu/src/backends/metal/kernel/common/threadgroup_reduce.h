#pragma once

#include <metal_stdlib>
#include "thread_context.h"

using namespace metal;

template <typename T>
struct SimdReduceSum {
  using value_type = T;
  static constant constexpr T identity = static_cast<T>(0);
  static T simd_reduce(T x) { return simd_sum(x); }
};

template <typename T>
struct SimdReduceMax {
  using value_type = T;
  static constant constexpr T identity = numeric_limits<T>::has_infinity
                                             ? -numeric_limits<T>::infinity()
                                             : numeric_limits<T>::lowest();
  static T simd_reduce(T x) { return simd_max(x); }
};

template <typename T>
struct SimdReduceMin {
  using value_type = T;
  static constant constexpr T identity = numeric_limits<T>::has_infinity
                                             ? numeric_limits<T>::infinity()
                                             : numeric_limits<T>::max();
  static T simd_reduce(T x) { return simd_min(x); }
};

template <typename Op, ushort BLOCK_SIZE>
static typename Op::value_type threadgroup_cooperative_reduce(
    typename Op::value_type value,
    threadgroup typename Op::value_type* shared,
    const thread ThreadContext& thread_context
) {
  // Phase 1: reduce within each simdgroup
  typename Op::value_type local = Op::simd_reduce(value);
  if (thread_context.simdgroup_index == 0) {
    shared[thread_context.threadgroup_index] = local;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Phase 2: first simdgroup reduces across simdgropus
  if (thread_context.threadgroup_index == 0) {
    typename Op::value_type total =
        thread_context.simdgroup_index <
                (BLOCK_SIZE + thread_context.simdgroup_size - 1) /
                    thread_context.simdgroup_size
            ? shared[thread_context.simdgroup_index]
            : Op::identity;
    total = Op::simd_reduce(total);

    if (thread_context.simdgroup_index == 0) {
      shared[0] = total;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Phase 3: broadcast
  const typename Op::value_type result = shared[0];

  threadgroup_barrier(mem_flags::mem_threadgroup);

  return result;
}
