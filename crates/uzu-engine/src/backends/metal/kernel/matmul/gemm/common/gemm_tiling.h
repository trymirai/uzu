#pragma once

#include "../../../generated/gemm.h"

namespace uzu {
namespace gemm {

constexpr uint gemm_tiling_block_m(GemmTiling t) {
  return t == GemmTiling::Tile8x32x32_Simdgroups1x1       ? 8
         : t == GemmTiling::Tile64x32x32_Simdgroups2x2    ? 64
         : t == GemmTiling::Tile64x64x16_Simdgroups2x2    ? 64
         : t == GemmTiling::Tile64x64x32_Simdgroups2x2    ? 64
         : t == GemmTiling::Tile32x32x32_Simdgroups2x2    ? 32
         : t == GemmTiling::Tile16x32x256_Simdgroups1x1   ? 16
         : t == GemmTiling::Tile16x32x256_Simdgroups1x1x4 ? 16
         : t == GemmTiling::Tile16x128x256_Simdgroups1x4  ? 16
         : t == GemmTiling::Tile32x64x256_Simdgroups2x2   ? 32
         : t == GemmTiling::Tile64x32x256_Simdgroups4x1   ? 64
         : t == GemmTiling::Tile64x64x256_Simdgroups2x2   ? 64
         : t == GemmTiling::Tile128x128x256_Simdgroups4x4 ? 128
         : t == GemmTiling::Tile8x32x64_Simdgroups1x2x4   ? 8
         : t == GemmTiling::Tile8x32x128_Simdgroups1x2x4  ? 8
                                                          : 0;
}

constexpr uint gemm_tiling_block_n(GemmTiling t) {
  return t == GemmTiling::Tile8x32x32_Simdgroups1x1       ? 32
         : t == GemmTiling::Tile64x32x32_Simdgroups2x2    ? 32
         : t == GemmTiling::Tile64x64x16_Simdgroups2x2    ? 64
         : t == GemmTiling::Tile64x64x32_Simdgroups2x2    ? 64
         : t == GemmTiling::Tile32x32x32_Simdgroups2x2    ? 32
         : t == GemmTiling::Tile16x32x256_Simdgroups1x1   ? 32
         : t == GemmTiling::Tile16x32x256_Simdgroups1x1x4 ? 32
         : t == GemmTiling::Tile16x128x256_Simdgroups1x4  ? 128
         : t == GemmTiling::Tile32x64x256_Simdgroups2x2   ? 64
         : t == GemmTiling::Tile64x32x256_Simdgroups4x1   ? 32
         : t == GemmTiling::Tile64x64x256_Simdgroups2x2   ? 64
         : t == GemmTiling::Tile128x128x256_Simdgroups4x4 ? 128
         : t == GemmTiling::Tile8x32x64_Simdgroups1x2x4   ? 32
         : t == GemmTiling::Tile8x32x128_Simdgroups1x2x4  ? 32
                                                          : 0;
}

constexpr uint gemm_tiling_block_k(GemmTiling t) {
  return t == GemmTiling::Tile8x32x32_Simdgroups1x1       ? 32
         : t == GemmTiling::Tile64x32x32_Simdgroups2x2    ? 32
         : t == GemmTiling::Tile64x64x16_Simdgroups2x2    ? 16
         : t == GemmTiling::Tile64x64x32_Simdgroups2x2    ? 32
         : t == GemmTiling::Tile32x32x32_Simdgroups2x2    ? 32
         : t == GemmTiling::Tile16x32x256_Simdgroups1x1   ? 256
         : t == GemmTiling::Tile16x32x256_Simdgroups1x1x4 ? 256
         : t == GemmTiling::Tile16x128x256_Simdgroups1x4  ? 256
         : t == GemmTiling::Tile32x64x256_Simdgroups2x2   ? 256
         : t == GemmTiling::Tile64x32x256_Simdgroups4x1   ? 256
         : t == GemmTiling::Tile64x64x256_Simdgroups2x2   ? 256
         : t == GemmTiling::Tile128x128x256_Simdgroups4x4 ? 256
         : t == GemmTiling::Tile8x32x64_Simdgroups1x2x4   ? 64
         : t == GemmTiling::Tile8x32x128_Simdgroups1x2x4  ? 128
                                                          : 0;
}

constexpr uint gemm_tiling_simdgroups_per_row(GemmTiling t) {
  return t == GemmTiling::Tile8x32x32_Simdgroups1x1       ? 1
         : t == GemmTiling::Tile64x32x32_Simdgroups2x2    ? 2
         : t == GemmTiling::Tile64x64x16_Simdgroups2x2    ? 2
         : t == GemmTiling::Tile64x64x32_Simdgroups2x2    ? 2
         : t == GemmTiling::Tile32x32x32_Simdgroups2x2    ? 2
         : t == GemmTiling::Tile16x32x256_Simdgroups1x1   ? 1
         : t == GemmTiling::Tile16x32x256_Simdgroups1x1x4 ? 1
         : t == GemmTiling::Tile16x128x256_Simdgroups1x4  ? 1
         : t == GemmTiling::Tile32x64x256_Simdgroups2x2   ? 2
         : t == GemmTiling::Tile64x32x256_Simdgroups4x1   ? 4
         : t == GemmTiling::Tile64x64x256_Simdgroups2x2   ? 2
         : t == GemmTiling::Tile128x128x256_Simdgroups4x4 ? 4
         : t == GemmTiling::Tile8x32x64_Simdgroups1x2x4   ? 1
         : t == GemmTiling::Tile8x32x128_Simdgroups1x2x4  ? 1
                                                          : 0;
}

constexpr uint gemm_tiling_simdgroups_per_column(GemmTiling t) {
  return t == GemmTiling::Tile8x32x32_Simdgroups1x1       ? 1
         : t == GemmTiling::Tile64x32x32_Simdgroups2x2    ? 2
         : t == GemmTiling::Tile64x64x16_Simdgroups2x2    ? 2
         : t == GemmTiling::Tile64x64x32_Simdgroups2x2    ? 2
         : t == GemmTiling::Tile32x32x32_Simdgroups2x2    ? 2
         : t == GemmTiling::Tile16x32x256_Simdgroups1x1   ? 1
         : t == GemmTiling::Tile16x32x256_Simdgroups1x1x4 ? 1
         : t == GemmTiling::Tile16x128x256_Simdgroups1x4  ? 4
         : t == GemmTiling::Tile32x64x256_Simdgroups2x2   ? 2
         : t == GemmTiling::Tile64x32x256_Simdgroups4x1   ? 1
         : t == GemmTiling::Tile64x64x256_Simdgroups2x2   ? 2
         : t == GemmTiling::Tile128x128x256_Simdgroups4x4 ? 4
         : t == GemmTiling::Tile8x32x64_Simdgroups1x2x4   ? 2
         : t == GemmTiling::Tile8x32x128_Simdgroups1x2x4  ? 2
                                                          : 0;
}

constexpr uint gemm_tiling_simdgroups_per_k(GemmTiling t) {
  return t == GemmTiling::Tile16x32x256_Simdgroups1x1x4  ? 4
         : t == GemmTiling::Tile8x32x64_Simdgroups1x2x4  ? 4
         : t == GemmTiling::Tile8x32x128_Simdgroups1x2x4 ? 4
                                                         : 1;
}

} // namespace gemm
} // namespace uzu
