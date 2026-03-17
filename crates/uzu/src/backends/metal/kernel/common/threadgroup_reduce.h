#pragma once

#include "thread_context.h"
#include <metal_stdlib>
using namespace metal;

template <ushort BLOCK_SIZE, typename T>
static T threadgroup_raking_prefix_exclusive_sum(
    T value,
    threadgroup T* shared,
    const ushort lid
) {
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

template <ushort BLOCK_SIZE, typename T>
static T threadgroup_raking_reduce_sum(
    T value,
    threadgroup T* shared,
    const ushort lid
) {
  shared[lid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lid < 32) {
    const short values_per_thread = BLOCK_SIZE / 32;
    const short first_index = lid * values_per_thread;

    T thread_sum = shared[first_index];
    for (short i = first_index + 1; i < first_index + values_per_thread; i++) {
      thread_sum += shared[i];
    }

    T total_sum = simd_sum(thread_sum);

    if (lid == 0) {
      shared[0] = total_sum;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const T result = shared[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}

template <ushort BLOCK_SIZE, typename T>
static T threadgroup_raking_reduce_max(
    T value,
    threadgroup T* shared,
    const ushort lid
) {
  shared[lid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lid < 32) {
    const short values_per_thread = BLOCK_SIZE / 32;
    const short first_index = lid * values_per_thread;

    T thread_max = shared[first_index];
    for (short i = first_index + 1; i < first_index + values_per_thread; i++) {
      thread_max = max(thread_max, shared[i]);
    }

    T total_max = simd_max(thread_max);

    if (lid == 0) {
      shared[0] = total_max;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const T result = shared[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}

template <ushort BLOCK_SIZE, typename T>
static T threadgroup_raking_reduce_min(
    T value,
    threadgroup T* shared,
    const ushort lid
) {
  shared[lid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lid < 32) {
    const short values_per_thread = BLOCK_SIZE / 32;
    const short first_index = lid * values_per_thread;

    T thread_min = shared[first_index];
    for (short i = first_index + 1; i < first_index + values_per_thread; i++) {
      thread_min = min(thread_min, shared[i]);
    }

    T total_min = simd_min(thread_min);

    if (lid == 0) {
      shared[0] = total_min;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const T result = shared[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}

template <ushort BLOCK_SIZE, typename T>
static T threadgroup_cooperative_reduce_sum(
    T value,
    threadgroup T* shared,
    const ushort lid,
    const thread ThreadContext& simd
) {
  T local_sum = simd_sum(value);

  if (simd.simdgroup_index == 0) {
    shared[simd.threadgroup_index] = local_sum;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  T total_sum = T(0);
  const ushort num_simd_groups =
      (BLOCK_SIZE + simd.simdgroup_size - 1) / simd.simdgroup_size;
  if (lid < num_simd_groups) {
    total_sum = shared[lid];
  }
  total_sum = simd_sum(total_sum);

  if (lid == 0) {
    shared[0] = total_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const T result = shared[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}

template <ushort BLOCK_SIZE, typename T>
static T threadgroup_cooperative_reduce_max(
    T value,
    threadgroup T* shared,
    const ushort lid,
    const thread ThreadContext& simd
) {
  T local_max = simd_max(value);

  if (simd.simdgroup_index == 0) {
    shared[simd.threadgroup_index] = local_max;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  T total_max =
      lid < (BLOCK_SIZE + simd.simdgroup_size - 1) / simd.simdgroup_size
          ? shared[lid]
          : T(-INFINITY);
  total_max = simd_max(total_max);

  if (lid == 0) {
    shared[0] = total_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const T result = shared[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}

template <ushort BLOCK_SIZE, typename T>
static T threadgroup_cooperative_reduce_min(
    T value,
    threadgroup T* shared,
    const ushort lid,
    const thread ThreadContext& simd
) {
  T local_min = simd_min(value);

  if (simd.simdgroup_index == 0) {
    shared[simd.threadgroup_index] = local_min;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  T total_min =
      lid < (BLOCK_SIZE + simd.simdgroup_size - 1) / simd.simdgroup_size
          ? shared[lid]
          : T(INFINITY);
  total_min = simd_min(total_min);

  if (lid == 0) {
    shared[0] = total_min;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const T result = shared[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}
