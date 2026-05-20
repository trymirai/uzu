#pragma once

#include "../../../generated/gemm.h"

namespace uzu {
namespace gemm {

constexpr uint gemm_tiling_block_m(GemmTiling t) {
  return t == GemmTiling::T8x32x32_1x1      ? 8
         : t == GemmTiling::T64x32x32_2x2   ? 64
         : t == GemmTiling::T64x64x16_2x2   ? 64
         : t == GemmTiling::T64x64x32_2x2   ? 64
         : t == GemmTiling::T32x32x32_2x2   ? 32
         : t == GemmTiling::T32x64x32_2x2   ? 32
         : t == GemmTiling::T64x32x32_4x1   ? 64
         : t == GemmTiling::T128x128x32_4x4 ? 128
                                            : 0;
}

constexpr uint gemm_tiling_block_n(GemmTiling t) {
  return t == GemmTiling::T8x32x32_1x1      ? 32
         : t == GemmTiling::T64x32x32_2x2   ? 32
         : t == GemmTiling::T64x64x16_2x2   ? 64
         : t == GemmTiling::T64x64x32_2x2   ? 64
         : t == GemmTiling::T32x32x32_2x2   ? 32
         : t == GemmTiling::T32x64x32_2x2   ? 64
         : t == GemmTiling::T64x32x32_4x1   ? 32
         : t == GemmTiling::T128x128x32_4x4 ? 128
                                            : 0;
}

constexpr uint gemm_tiling_block_k(GemmTiling t) {
  return t == GemmTiling::T8x32x32_1x1      ? 32
         : t == GemmTiling::T64x32x32_2x2   ? 32
         : t == GemmTiling::T64x64x16_2x2   ? 16
         : t == GemmTiling::T64x64x32_2x2   ? 32
         : t == GemmTiling::T32x32x32_2x2   ? 32
         : t == GemmTiling::T32x64x32_2x2   ? 32
         : t == GemmTiling::T64x32x32_4x1   ? 32
         : t == GemmTiling::T128x128x32_4x4 ? 32
                                            : 0;
}

constexpr uint gemm_tiling_simdgroups_per_row(GemmTiling t) {
  return t == GemmTiling::T8x32x32_1x1      ? 1
         : t == GemmTiling::T64x32x32_2x2   ? 2
         : t == GemmTiling::T64x64x16_2x2   ? 2
         : t == GemmTiling::T64x64x32_2x2   ? 2
         : t == GemmTiling::T32x32x32_2x2   ? 2
         : t == GemmTiling::T32x64x32_2x2   ? 2
         : t == GemmTiling::T64x32x32_4x1   ? 4
         : t == GemmTiling::T128x128x32_4x4 ? 4
                                            : 0;
}

constexpr uint gemm_tiling_simdgroups_per_column(GemmTiling t) {
  return t == GemmTiling::T8x32x32_1x1      ? 1
         : t == GemmTiling::T64x32x32_2x2   ? 2
         : t == GemmTiling::T64x64x16_2x2   ? 2
         : t == GemmTiling::T64x64x32_2x2   ? 2
         : t == GemmTiling::T32x32x32_2x2   ? 2
         : t == GemmTiling::T32x64x32_2x2   ? 2
         : t == GemmTiling::T64x32x32_4x1   ? 1
         : t == GemmTiling::T128x128x32_4x4 ? 4
                                            : 0;
}

} // namespace gemm
} // namespace uzu
