#pragma once

#define METAL_CONST static constant constexpr
#define METAL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define METAL_PRAGMA_NO_UNROLL _Pragma("clang loop unroll(disable)")
