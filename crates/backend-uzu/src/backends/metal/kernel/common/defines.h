#pragma once

#define METAL_CONST static constant constexpr
#define METAL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define METAL_PRAGMA_NO_UNROLL _Pragma("clang loop unroll(disable)")
#define METAL_SIMD_SIZE 32

METAL_FUNC int pow2(int n) { return 1 << n; }

template <typename T, typename U>
METAL_FUNC T div_ceil(T dividend, U divisor) {
  return (dividend + T(divisor) - T(1)) / T(divisor);
}
