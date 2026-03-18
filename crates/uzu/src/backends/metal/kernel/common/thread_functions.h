#pragma once

#include <metal_stdlib>
using namespace metal;

template <ushort LENGTH, typename T>
static inline T thread_prefix_inclusive_sum(thread T (&values)[LENGTH]) {
  for (ushort i = 1; i < LENGTH; i++) {
    values[i] += values[i - 1];
  }
  return values[LENGTH - 1];
}

template <ushort LENGTH, typename T>
static inline T thread_prefix_inclusive_sum(threadgroup T* values) {
  for (ushort i = 1; i < LENGTH; i++) {
    values[i] += values[i - 1];
  }
  return values[LENGTH - 1];
}

template <ushort LENGTH, typename T>
static inline T thread_prefix_exclusive_sum(thread T (&values)[LENGTH]) {
  T inclusive_prefix = thread_prefix_inclusive_sum<LENGTH>(values);
  for (ushort i = LENGTH - 1; i > 0; i--) {
    values[i] = values[i - 1];
  }
  values[0] = 0;
  return inclusive_prefix;
}

template <ushort LENGTH, typename T>
static inline T thread_prefix_exclusive_sum(threadgroup T* values) {
  T inclusive_prefix = thread_prefix_inclusive_sum<LENGTH>(values);
  for (ushort i = LENGTH - 1; i > 0; i--) {
    values[i] = values[i - 1];
  }
  values[0] = 0;
  return inclusive_prefix;
}

template <ushort LENGTH, typename T>
static inline void thread_uniform_add(thread T (&values)[LENGTH], T uni) {
  for (ushort i = 0; i < LENGTH; i++) {
    values[i] += uni;
  }
}

template <ushort LENGTH, typename T>
static inline void thread_uniform_add(threadgroup T* values, T uni) {
  for (ushort i = 0; i < LENGTH; i++) {
    values[i] += uni;
  }
}
