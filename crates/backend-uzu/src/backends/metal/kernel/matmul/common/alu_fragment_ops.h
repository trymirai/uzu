#pragma once

#include "../../common/integral_constant.h"
#include "../../common/thread_context.h"
#include "defines.h"

using namespace metal;

namespace uzu {
namespace matmul {

///////////////////////////////////////////////////////////////////////////////
// AluFragmentOps - stateless traits/ops for 8x8 ALU matmul (simdgroup_matrix)
///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct AluFragmentOps {
  METAL_CONST ushort FRAGMENT_ROWS = 8;
  METAL_CONST ushort FRAGMENT_COLS = 8;
  METAL_CONST ushort ELEMENTS_PER_THREAD =
      (FRAGMENT_ROWS * FRAGMENT_COLS) / METAL_SIMD_SIZE;
  METAL_CONST ushort THREAD_ELEMENT_ROWS = 1;
  METAL_CONST ushort THREAD_ELEMENT_COLS = 2;
  METAL_CONST ushort THREAD_ELEMENT_ROW_STRIDE = 1;

  static_assert(
      THREAD_ELEMENT_ROWS * THREAD_ELEMENT_COLS == ELEMENTS_PER_THREAD,
      "AluFragmentOps shape is not consistent with element count"
  );

  typedef metal::simdgroup_matrix<T, FRAGMENT_ROWS, FRAGMENT_COLS>
      SimdgroupMatrixType;

  template <typename U>
  using ThreadVector = metal::vec<U, ELEMENTS_PER_THREAD>;

  METAL_FUNC static constexpr void multiply_accumulate(
      thread ThreadVector<T>& D,
      thread ThreadVector<T>& A,
      thread ThreadVector<T>& B,
      thread ThreadVector<T>& C
  ) {
    SimdgroupMatrixType D_mat;
    SimdgroupMatrixType A_mat;
    SimdgroupMatrixType B_mat;
    SimdgroupMatrixType C_mat;

    reinterpret_cast<thread ThreadVector<T>&>(A_mat.thread_elements()) = A;
    reinterpret_cast<thread ThreadVector<T>&>(B_mat.thread_elements()) = B;
    reinterpret_cast<thread ThreadVector<T>&>(C_mat.thread_elements()) = C;

    simdgroup_multiply_accumulate(D_mat, A_mat, B_mat, C_mat);

    D = reinterpret_cast<thread ThreadVector<T>&>(D_mat.thread_elements());
  }

  template <class OutputTile, class LeftTile, class RightTile>
  METAL_FUNC static void tile_matmul(
      thread OutputTile& output,
      thread LeftTile& left,
      thread RightTile& right
  ) {
    constexpr ushort tile_m = OutputTile::TILE_ROWS;
    constexpr ushort tile_n = OutputTile::TILE_COLS;
    constexpr ushort tile_k = LeftTile::TILE_COLS;

    static_assert(
        tile_m == LeftTile::TILE_ROWS,
        "tile matmul: M dimensions do not match"
    );
    static_assert(
        tile_n == RightTile::TILE_COLS,
        "tile matmul: N dimensions do not match"
    );
    static_assert(
        tile_k == RightTile::TILE_ROWS,
        "tile matmul: K dimensions do not match"
    );

    METAL_PRAGMA_UNROLL
    for (ushort row = 0; row < tile_m; ++row) {
      METAL_PRAGMA_UNROLL
      for (ushort col = 0; col < tile_n; ++col) {
        const ushort column_serpentine = (row % 2) ? (tile_n - 1 - col) : col;
        METAL_PRAGMA_UNROLL
        for (ushort k = 0; k < tile_k; ++k) {
          multiply_accumulate(
              output.fragment_at(row, column_serpentine),
              left.fragment_at(row, k),
              right.fragment_at(k, column_serpentine),
              output.fragment_at(row, column_serpentine)
          );
        }
      }
    }
  }
};

} // namespace matmul
} // namespace uzu
