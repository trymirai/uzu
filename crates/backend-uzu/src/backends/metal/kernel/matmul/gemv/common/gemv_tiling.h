#pragma once

#include "../../../generated/matmul.h"

// Device-side geometry accessors for GemvTiling. The kernel passes the tiling
// as a function-constant specialization (not a template parameter), so these
// resolve at pipeline-creation time, never at C++ compile time — fine, since
// nothing here sizes a threadgroup array.
namespace uzu {
namespace matmul {

constexpr uint gemv_tiling_tg_simd_rows(GemvTiling t) {
  return t == GemvTiling::Standard           ? 4
         : t == GemvTiling::StandardNarrow   ? 4
         : t == GemvTiling::Wide             ? 8
         : t == GemvTiling::WideNarrow       ? 8
         : t == GemvTiling::SmallInput       ? 1
         : t == GemvTiling::SmallInputNarrow ? 1
         : t == GemvTiling::SplitInput       ? 1
         : t == GemvTiling::SplitInputNarrow ? 1
                                             : 0;
}

constexpr uint gemv_tiling_tg_simd_cols(GemvTiling t) {
  return t == GemvTiling::SplitInput         ? 8
         : t == GemvTiling::SplitInputNarrow ? 8
                                             : 1;
}

constexpr uint gemv_tiling_sg_thread_rows(GemvTiling t) {
  return t == GemvTiling::SmallInput         ? 8
         : t == GemvTiling::SmallInputNarrow ? 8
                                             : 1;
}

constexpr uint gemv_tiling_sg_thread_cols(GemvTiling t) {
  return t == GemvTiling::SmallInput         ? 4
         : t == GemvTiling::SmallInputNarrow ? 4
                                             : 32;
}

constexpr uint gemv_tiling_thread_out_rows(GemvTiling t) {
  return t == GemvTiling::StandardNarrow       ? 1
         : t == GemvTiling::WideNarrow         ? 1
         : t == GemvTiling::SmallInputNarrow   ? 1
         : t == GemvTiling::SplitInputNarrow   ? 1
                                               : 4;
}

constexpr uint gemv_tiling_thread_out_cols(GemvTiling t) {
  (void)t;
  return 4;
}

} // namespace matmul
} // namespace uzu
