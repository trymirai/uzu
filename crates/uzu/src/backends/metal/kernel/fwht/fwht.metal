#include <metal_stdlib>
#include "../definitions.metal"

#include "fwht.h"

// Full-vector in-place Hadamard transform.
// Ported from MLX's hadamard_n kernel.
// One threadgroup processes one row of length N.
// Uses threadgroup memory for inter-thread communication.

#define MAX_RADIX 16

template <typename T, int N>
VARIANTS(T, half, float, bfloat)
VARIANTS(N, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
KERNEL(Fwht)(
    device T* data,
    constant uint& batch_size,
    constant float& scale,
    threadgroup T shared_buf[N],
    const uint batch_idx GROUPS(batch_size),
    const uint tid THREADS(512)
) {
  constexpr short max_radix = MAX_RADIX;
  constexpr short num_threads = N / max_radix;
  if (tid >= num_threads) return;
  constexpr short logN = __builtin_ctz(N);
  constexpr short logR = __builtin_ctz(max_radix);
  constexpr short num_steps = logN / logR;
  constexpr short logFinal = logN % logR;
  constexpr short final_radix = 1 << logFinal;

  device T* row = data + batch_idx * N;
  short i = tid;

  // Load row into threadgroup memory
  STEEL_PRAGMA_UNROLL
  for (short j = 0; j < max_radix; j++) {
    shared_buf[j * num_threads + i] = row[j * num_threads + i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Main radix stages
  float x[max_radix];
  short h = 1;

  STEEL_PRAGMA_UNROLL
  for (short s = 0; s < num_steps; s++) {
    short k = i & (h - 1);
    short j = ((i - k) << logR) + k;

    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < max_radix; r++) {
      x[r] = float(shared_buf[j + h * r]);
    }

    radix_func<max_radix>(x);

    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < max_radix; r++) {
      shared_buf[j + h * r] = T(x[r]);
    }

    h <<= logR;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Final partial radix stage (when logN is not a multiple of logR)
  IF_CONSTEXPR(final_radix > 1) {
    STEEL_PRAGMA_UNROLL
    for (int t = 0; t < max_radix / final_radix; t++) {
      short index = i + t * num_threads;
      short k = index & (h - 1);
      short j = ((index - k) << logFinal) + k;

      STEEL_PRAGMA_UNROLL
      for (short r = 0; r < final_radix; r++) {
        x[r] = float(shared_buf[j + h * r]);
      }

      radix_func<final_radix>(x);

      STEEL_PRAGMA_UNROLL
      for (short r = 0; r < final_radix; r++) {
        shared_buf[j + h * r] = T(x[r]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write back to device memory with scale
  STEEL_PRAGMA_UNROLL
  for (short j = 0; j < max_radix; j++) {
    row[j * num_threads + i] = T(float(shared_buf[j * num_threads + i]) * scale);
  }
}
