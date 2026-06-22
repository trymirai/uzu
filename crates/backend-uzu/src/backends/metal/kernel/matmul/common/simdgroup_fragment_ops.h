#pragma once

#include "defines.h"
#include "simdgroup_multiply_accumulate.h"

using namespace metal;

namespace uzu {
namespace matmul {

// FragmentOps for the 8x8 simdgroup_matrix (pre-M5): lets Fragment<T,R,C,
// SimdgroupFragmentOps> share Fragment's load/store/for_each_element with the
// simdgroup backend. The 8x8 MMA + lane layout live in SimdgroupMultiplyAccumulate.
struct SimdgroupFragmentOps {
  METAL_CONST ushort FRAGMENT_ROWS = 8;
  METAL_CONST ushort FRAGMENT_COLS = 8;

  // FragmentOps conformance (Fragment::transfer / for_each_element rely on these).
  METAL_CONST ushort ELEMENTS_PER_THREAD = (FRAGMENT_ROWS * FRAGMENT_COLS) / METAL_SIMD_SIZE;
  METAL_CONST ushort THREAD_ELEMENT_ROWS = 1;
  METAL_CONST ushort THREAD_ELEMENT_COLS = 2;
  METAL_CONST ushort THREAD_ELEMENT_ROW_STRIDE = 8;

  template <typename U>
  using ThreadVector = typename metal::vec<U, ELEMENTS_PER_THREAD>;

  // output += left @ right, sub-tile by sub-tile. Serpentine column order
  // matches the chain kernel so register reuse is identical. Transpose is
  // handled at load time -> only the non-transposed form is supported.
  template <bool transpose_a, bool transpose_b, class OutputTile, class LeftTile, class RightTile>
  METAL_FUNC static void tile_matmul(thread OutputTile& output, thread LeftTile& left, thread RightTile& right) {
    static_assert(!transpose_a && !transpose_b, "SimdgroupFragmentOps::tile_matmul: transpose not supported");
    constexpr ushort tile_m = OutputTile::TILE_ROWS;
    constexpr ushort tile_n = OutputTile::TILE_COLS;
    constexpr ushort tile_k = LeftTile::TILE_COLS;
    static_assert(LeftTile::TILE_ROWS == tile_m, "tile matmul: M dimensions do not match");
    static_assert(RightTile::TILE_COLS == tile_n, "tile matmul: N dimensions do not match");
    static_assert(RightTile::TILE_ROWS == tile_k, "tile matmul: K dimensions do not match");
    using Sg = SimdgroupMultiplyAccumulate<typename OutputTile::ElementType, FRAGMENT_ROWS, FRAGMENT_COLS>;

    METAL_PRAGMA_UNROLL
    for (ushort m = 0; m < tile_m; ++m) {
      METAL_PRAGMA_UNROLL
      for (ushort n = 0; n < tile_n; ++n) {
        const ushort col = (m % 2) ? (tile_n - 1 - n) : n;
        METAL_PRAGMA_UNROLL
        for (ushort k = 0; k < tile_k; ++k) {
          Sg::multiply_accumulate(
              output.fragment_at(m, col),
              left.fragment_at(m, k),
              right.fragment_at(k, col),
              output.fragment_at(m, col)
          );
        }
      }
    }
  }
};

} // namespace matmul
} // namespace uzu
