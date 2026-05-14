#include <metal_stdlib>

#include "../common/defines.h"
#include "../common/dsl.h"

// Online row-wise softmax (MLX-style: simd intrinsics + 2-pass online
// algorithm). `has_sinks` seeds thread 0 with (sinks[outer_index], 1) as a
// virtual element.

#define SOFTMAX_THREADS 256
// Number of consecutive elements each thread loads per iteration. Vectorizing
// the per-thread loop this way improves memory coalescing on Apple Silicon.
#define SOFTMAX_ELEMENTS_PER_THREAD 4

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(Softmax)(
    device T* values,
    const device float* sinks OPTIONAL(has_sinks),
    const constant uint& row_length,
    const constant uint& outer_dim,
    const constant uint& batch_dim,
    threadgroup float shared_max[METAL_SIMD_SIZE],
    threadgroup float shared_norm[METAL_SIMD_SIZE],
    const bool has_sinks SPECIALIZE,
    const uint outer_index GROUPS(outer_dim),
    const uint batch_index GROUPS(batch_dim),
    const uint tid THREADS(SOFTMAX_THREADS)
) {
  const uint lane = tid & (METAL_SIMD_SIZE - 1);
  const uint simdgroup_id = tid / METAL_SIMD_SIZE;
  const uint stride = SOFTMAX_THREADS * SOFTMAX_ELEMENTS_PER_THREAD;
  device T* row = values + (outer_index * batch_dim + batch_index) * row_length;

  float maxval = (has_sinks && tid == 0) ? sinks[outer_index] : -FLT_MAX;
  float norm = (has_sinks && tid == 0) ? 1.0f : 0.0f;

  if (simdgroup_id == 0) {
    shared_max[lane] = -FLT_MAX;
    shared_norm[lane] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Pass 1: online (max, normalizer) accumulation over the row.
  for (uint base = tid * SOFTMAX_ELEMENTS_PER_THREAD; base < row_length;
       base += stride) {
    float vals[SOFTMAX_ELEMENTS_PER_THREAD];
    for (uint i = 0; i < SOFTMAX_ELEMENTS_PER_THREAD; i++) {
      vals[i] = (base + i < row_length) ? float(row[base + i]) : -FLT_MAX;
    }
    float prev_max = maxval;
    for (uint i = 0; i < SOFTMAX_ELEMENTS_PER_THREAD; i++) {
      maxval = max(maxval, vals[i]);
    }
    norm *= fast::exp(prev_max - maxval);
    for (uint i = 0; i < SOFTMAX_ELEMENTS_PER_THREAD; i++) {
      norm += fast::exp(vals[i] - maxval);
    }
  }

  // Combine across the simdgroup, then across simdgroups via shared memory.
  float prev_max = maxval;
  maxval = simd_max(maxval);
  norm *= fast::exp(prev_max - maxval);
  norm = simd_sum(norm);

  if (lane == 0) {
    shared_max[simdgroup_id] = maxval;
    shared_norm[simdgroup_id] = norm;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup_id == 0) {
    float simdgroup_max = shared_max[lane];
    float simdgroup_norm = shared_norm[lane];
    float global_max = simd_max(simdgroup_max);
    simdgroup_norm *= fast::exp(simdgroup_max - global_max);
    float global_norm = simd_sum(simdgroup_norm);
    if (lane == 0) {
      shared_max[0] = global_max;
      shared_norm[0] = global_norm;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float final_max = shared_max[0];
  const float inv_norm = 1.0f / shared_norm[0];

  // Pass 2: normalize.
  for (uint base = tid * SOFTMAX_ELEMENTS_PER_THREAD; base < row_length;
       base += stride) {
    for (uint i = 0; i < SOFTMAX_ELEMENTS_PER_THREAD; i++) {
      if (base + i < row_length) {
        row[base + i] =
            T(fast::exp(float(row[base + i]) - final_max) * inv_norm);
      }
    }
  }
}
