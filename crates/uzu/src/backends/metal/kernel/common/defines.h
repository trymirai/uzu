#pragma once

#define METAL_CONST static constant constexpr
#define METAL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define METAL_SIMD_SIZE 32

METAL_FUNC int pow2(int n) { return 1 << n; }
