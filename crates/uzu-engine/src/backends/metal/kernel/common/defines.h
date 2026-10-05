#pragma once

#define UZU_CONST static constant constexpr
#define METAL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define METAL_PRAGMA_NO_UNROLL _Pragma("clang loop unroll(disable)")
#define METAL_SIMD_SIZE 32

template <typename T, typename U>
METAL_FUNC T div_ceil(T dividend, U divisor) {
  return (dividend + T(divisor) - T(1)) / T(divisor);
}

// Element-wise conversion between vector types with the same number of lanes. In MSL this is the
// converting constructor; other front ends (the AMDGPU compatibility layer) provide their own.
#ifndef UZU_HAS_VECTOR_CAST
template <typename To, typename From>
METAL_FUNC To vector_cast(From value) {
  return To(value);
}
#endif
