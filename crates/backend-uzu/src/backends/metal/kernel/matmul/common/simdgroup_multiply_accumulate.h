#pragma once

#include "../../common/integral_constant.h"
#include "../../common/thread_context.h"
#include "defines.h"

using namespace metal;

namespace uzu {
namespace matmul {

///////////////////////////////////////////////////////////////////////////////
// SimdgroupFragmentOps - stateless traits/ops for 8x8 simdgroup_matrix
///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct SimdgroupFragmentOps {
  METAL_CONST int FRAGMENT_ROWS = 8;
  METAL_CONST int FRAGMENT_COLS = 8;
  METAL_CONST int ELEMENTS_PER_THREAD =
      (FRAGMENT_ROWS * FRAGMENT_COLS) / METAL_SIMD_SIZE;
  METAL_CONST int THREAD_ELEMENT_ROWS = 1;
  METAL_CONST int THREAD_ELEMENT_COLS = 2;

  static_assert(
      THREAD_ELEMENT_ROWS * THREAD_ELEMENT_COLS == ELEMENTS_PER_THREAD,
      "SimdgroupMultiplyAccumulate shape is not consistent with element count"
  );

  typedef metal::simdgroup_matrix<T, FRAGMENT_ROWS, FRAGMENT_COLS>
      SimdgroupMatrixType;

  template <typename U>
  using ThreadVector = metal::vec<U, ELEMENTS_PER_THREAD>;

  METAL_FUNC static constexpr short2 get_position(
      const thread ThreadContext& thread_context
  ) {
    const ushort simdgroup_index = ushort(thread_context.simdgroup_index);
    const short quad_index = simdgroup_index / 4;
    const short position_row = (quad_index & 4) + (simdgroup_index / 2) % 4;
    const short position_col =
        ((quad_index & 2) + simdgroup_index % 2) * THREAD_ELEMENT_COLS;
    return short2{position_col, position_row};
  }

  template <
      typename SourcePointerType,
      typename RowStride,
      typename ColStride,
      typename RowOffset = Int<0>,
      typename ColOffset = Int<0>>
  METAL_FUNC static constexpr void load(
      thread ThreadVector<T>& destination,
      SourcePointerType source,
      RowStride row_stride,
      ColStride col_stride,
      RowOffset row_offset,
      ColOffset col_offset,
      const thread ThreadContext& thread_context
  ) {
    const short2 position = get_position(thread_context);
    source += position.y * row_stride + position.x * col_stride;

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < THREAD_ELEMENT_ROWS; i++) {
      METAL_PRAGMA_UNROLL
      for (ushort j = 0; j < THREAD_ELEMENT_COLS; j++) {
        destination[i * THREAD_ELEMENT_COLS + j] = static_cast<T>(
            source
                [(row_offset + i) * row_stride + (col_offset + j) * col_stride]
        );
      }
    }
  }

  template <
      typename SourcePointerType,
      typename RowStride,
      typename ColStride,
      typename RowLimit,
      typename ColLimit,
      typename RowOffset = Int<0>,
      typename ColOffset = Int<0>>
  METAL_FUNC static constexpr void load_safe(
      thread ThreadVector<T>& destination,
      SourcePointerType source,
      RowStride row_stride,
      ColStride col_stride,
      RowLimit row_limit,
      ColLimit col_limit,
      RowOffset row_offset,
      ColOffset col_offset,
      const thread ThreadContext& thread_context
  ) {
    const short2 position = get_position(thread_context);
    source += position.y * row_stride + position.x * col_stride;
    const auto local_row_limit = row_limit - position.y;
    const auto local_col_limit = col_limit - position.x;

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < THREAD_ELEMENT_ROWS; i++) {
      METAL_PRAGMA_UNROLL
      for (ushort j = 0; j < THREAD_ELEMENT_COLS; j++) {
        if ((row_offset + i) < local_row_limit &&
            (col_offset + j) < local_col_limit) {
          destination[i * THREAD_ELEMENT_COLS + j] =
              static_cast<T>(source
                                 [(row_offset + i) * row_stride +
                                  (col_offset + j) * col_stride]);
        } else {
          destination[i * THREAD_ELEMENT_COLS + j] = T(0);
        }
      }
    }
  }

  template <
      typename DestinationPointerType,
      typename RowStride,
      typename ColStride,
      typename RowOffset = Int<0>,
      typename ColOffset = Int<0>>
  METAL_FUNC static constexpr void store(
      const thread ThreadVector<T>& source,
      DestinationPointerType destination,
      RowStride row_stride,
      ColStride col_stride,
      RowOffset row_offset,
      ColOffset col_offset,
      const thread ThreadContext& thread_context
  ) {
    using U = PointerElementType<DestinationPointerType>;

    const short2 position = get_position(thread_context);
    destination += position.y * row_stride + position.x * col_stride;

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < THREAD_ELEMENT_ROWS; i++) {
      METAL_PRAGMA_UNROLL
      for (ushort j = 0; j < THREAD_ELEMENT_COLS; j++) {
        destination
            [(row_offset + i) * row_stride + (col_offset + j) * col_stride] =
                static_cast<U>(source[i * THREAD_ELEMENT_COLS + j]);
      }
    }
  }

  template <
      typename DestinationPointerType,
      typename RowStride,
      typename ColStride,
      typename RowLimit,
      typename ColLimit,
      typename RowOffset = Int<0>,
      typename ColOffset = Int<0>>
  METAL_FUNC static constexpr void store_safe(
      const thread ThreadVector<T>& source,
      DestinationPointerType destination,
      RowStride row_stride,
      ColStride col_stride,
      RowLimit row_limit,
      ColLimit col_limit,
      RowOffset row_offset,
      ColOffset col_offset,
      const thread ThreadContext& thread_context
  ) {
    using U = PointerElementType<DestinationPointerType>;

    const short2 position = get_position(thread_context);
    destination += position.y * row_stride + position.x * col_stride;
    const auto local_row_limit = row_limit - position.y;
    const auto local_col_limit = col_limit - position.x;

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < THREAD_ELEMENT_ROWS; i++) {
      METAL_PRAGMA_UNROLL
      for (ushort j = 0; j < THREAD_ELEMENT_COLS; j++) {
        if ((row_offset + i) < local_row_limit &&
            (col_offset + j) < local_col_limit) {
          destination
              [(row_offset + i) * row_stride + (col_offset + j) * col_stride] =
                  static_cast<U>(source[i * THREAD_ELEMENT_COLS + j]);
        }
      }
    }
  }

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

  template <
      bool transpose_a,
      bool transpose_b,
      class OutputTile,
      class LeftTile,
      class RightTile>
  METAL_FUNC static void tile_matmul(
      thread OutputTile& output,
      thread LeftTile& left,
      thread RightTile& right
  ) {
    (void)transpose_a;
    (void)transpose_b;

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
