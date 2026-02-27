#ifndef fwht_h
#define fwht_h

#include <metal_stdlib>

using namespace metal;

#ifndef STEEL_PRAGMA_UNROLL
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#endif

// Thread-local Hadamard transform for 2^R elements.
// Ported from MLX (external/mlx/mlx/backend/metal/kernels/hadamard.h).
// Operates entirely in thread-private registers — no barriers needed.
// All loops are unrolled at compile time.
template <short R>
METAL_FUNC void radix_func(thread float* x) {
  constexpr short logR = __builtin_ctz(R);
  short h = 1;
  STEEL_PRAGMA_UNROLL
  for (short s = 0; s < logR; s++) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < R / 2; i++) {
      short k = i & (h - 1);
      short j = ((i - k) << 1) + k;
      float a = x[j];
      float b = x[j + h];
      x[j] = a + b;
      x[j + h] = a - b;
    }
    h <<= 1;
  }
}

#endif /* fwht_h */
