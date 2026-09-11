#pragma once

#include "defines.h"

using namespace metal;

namespace uzu {
namespace matmul {

UZU_CONST int SIMDGROUP_MMA_ROWS = 8;
UZU_CONST int SIMDGROUP_MMA_COLS = 8;

///////////////////////////////////////////////////////////////////////////////
// SimdgroupMMA - stateless traits/ops for 8x8 simdgroup_matrix
///////////////////////////////////////////////////////////////////////////////

template <typename AccumulatorT, int ROWS_, int COLS_, typename LeftT = AccumulatorT, typename RightT = AccumulatorT>
struct SimdgroupMMA {
  static_assert(ROWS_ == SIMDGROUP_MMA_ROWS, "Only 8 x 8 fragment matrices are currently supported");
  static_assert(COLS_ == SIMDGROUP_MMA_COLS, "Only 8 x 8 fragment matrices are currently supported");
};

template <typename AccumulatorT, typename LeftT, typename RightT>
struct SimdgroupMMA<AccumulatorT, SIMDGROUP_MMA_ROWS, SIMDGROUP_MMA_COLS, LeftT, RightT> {
  UZU_CONST int ROWS = SIMDGROUP_MMA_ROWS;
  UZU_CONST int COLS = SIMDGROUP_MMA_COLS;
  UZU_CONST int THREAD_ELEMENT_COUNT = (ROWS * COLS) / METAL_SIMD_SIZE;
  UZU_CONST int THREAD_ELEMENT_ROWS = 1;
  UZU_CONST int THREAD_ELEMENT_COLS = 2;

  static_assert(
      THREAD_ELEMENT_ROWS * THREAD_ELEMENT_COLS == THREAD_ELEMENT_COUNT,
      "SimdgroupMMA shape is not consistent with element count"
  );

  typedef metal::simdgroup_matrix<AccumulatorT, ROWS, COLS> AccumulatorMatrixType;
  typedef metal::simdgroup_matrix<LeftT, ROWS, COLS> LeftMatrixType;
  typedef metal::simdgroup_matrix<RightT, ROWS, COLS> RightMatrixType;
  typedef metal::vec<AccumulatorT, THREAD_ELEMENT_COUNT> AccumulatorThreadDataType;
  typedef metal::vec<LeftT, THREAD_ELEMENT_COUNT> LeftThreadDataType;
  typedef metal::vec<RightT, THREAD_ELEMENT_COUNT> RightThreadDataType;
  typedef AccumulatorMatrixType SimdgroupMatrixType;
  typedef AccumulatorThreadDataType ThreadDataType;

  METAL_FUNC static constexpr short2 get_lane_coordinates(ushort simd_lane_id [[thread_index_in_simdgroup]]) {
    const short quad_index = simd_lane_id / 4;
    const short lane_row = (quad_index & 4) + ((simd_lane_id / 2) % 4);
    const short lane_col = (quad_index & 2) * 2 + (simd_lane_id % 2) * 2;
    return short2{lane_col, lane_row};
  }

  template <typename SourcePointerType, typename RowStride, typename ColStride>
  METAL_FUNC static constexpr void load(
      thread ThreadDataType& destination,
      SourcePointerType source,
      RowStride row_stride,
      ColStride col_stride
  ) {
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < THREAD_ELEMENT_ROWS; i++) {
      METAL_PRAGMA_UNROLL
      for (ushort j = 0; j < THREAD_ELEMENT_COLS; j++) {
        destination[i * THREAD_ELEMENT_COLS + j] = static_cast<AccumulatorT>(source[i * row_stride + j * col_stride]);
      }
    }
  }

  template <
      typename SourcePointerType,
      typename RowStride,
      typename ColStride,
      typename RowLimit,
      typename ColLimit,
      typename RowOffset,
      typename ColOffset>
  METAL_FUNC static constexpr void load_safe(
      thread ThreadDataType& destination,
      SourcePointerType source,
      RowStride row_stride,
      ColStride col_stride,
      RowLimit row_limit,
      ColLimit col_limit,
      RowOffset row_offset = 0,
      ColOffset col_offset = 0
  ) {
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < THREAD_ELEMENT_ROWS; i++) {
      METAL_PRAGMA_UNROLL
      for (ushort j = 0; j < THREAD_ELEMENT_COLS; j++) {
        if ((row_offset + i) < row_limit && (col_offset + j) < col_limit) {
          destination[i * THREAD_ELEMENT_COLS + j] =
              static_cast<AccumulatorT>(source[(row_offset + i) * row_stride + (col_offset + j) * col_stride]);
        } else {
          destination[i * THREAD_ELEMENT_COLS + j] = AccumulatorT(0);
        }
      }
    }
  }

  template <typename DestinationPointerType, typename RowStride, typename ColStride>
  METAL_FUNC static constexpr void store(
      const thread ThreadDataType& source,
      DestinationPointerType destination,
      RowStride row_stride,
      ColStride col_stride
  ) {
    using U = PointerElementType<DestinationPointerType>;

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < THREAD_ELEMENT_ROWS; i++) {
      METAL_PRAGMA_UNROLL
      for (ushort j = 0; j < THREAD_ELEMENT_COLS; j++) {
        destination[i * row_stride + j * col_stride] = static_cast<U>(source[i * THREAD_ELEMENT_COLS + j]);
      }
    }
  }

  template <
      typename DestinationPointerType,
      typename RowStride,
      typename ColStride,
      typename RowLimit,
      typename ColLimit,
      typename RowOffset,
      typename ColOffset>
  METAL_FUNC static constexpr void store_safe(
      const thread ThreadDataType& source,
      DestinationPointerType destination,
      RowStride row_stride,
      ColStride col_stride,
      RowLimit row_limit,
      ColLimit col_limit,
      RowOffset row_offset = 0,
      ColOffset col_offset = 0
  ) {
    using U = PointerElementType<DestinationPointerType>;

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < THREAD_ELEMENT_ROWS; i++) {
      METAL_PRAGMA_UNROLL
      for (ushort j = 0; j < THREAD_ELEMENT_COLS; j++) {
        if ((row_offset + i) < row_limit && (col_offset + j) < col_limit) {
          destination[(row_offset + i) * row_stride + (col_offset + j) * col_stride] =
              static_cast<U>(source[i * THREAD_ELEMENT_COLS + j]);
        }
      }
    }
  }

  METAL_FUNC static constexpr void mma(
      thread AccumulatorThreadDataType& D,
      const thread LeftThreadDataType& A,
      const thread RightThreadDataType& B,
      const thread AccumulatorThreadDataType& C
  ) {
    AccumulatorMatrixType D_mat;
    LeftMatrixType A_mat;
    RightMatrixType B_mat;
    AccumulatorMatrixType C_mat;

    reinterpret_cast<thread LeftThreadDataType&>(A_mat.thread_elements()) = A;
    reinterpret_cast<thread RightThreadDataType&>(B_mat.thread_elements()) = B;
    reinterpret_cast<thread AccumulatorThreadDataType&>(C_mat.thread_elements()) = C;

    mma(D_mat, A_mat, B_mat, C_mat);

    D = reinterpret_cast<thread AccumulatorThreadDataType&>(D_mat.thread_elements());
  }

  METAL_FUNC static constexpr void mma(
      thread AccumulatorMatrixType& D,
      thread LeftMatrixType& A,
      thread RightMatrixType& B,
      thread AccumulatorMatrixType& C
  ) {
    simdgroup_multiply_accumulate(D, A, B, C);
  }
};

} // namespace matmul
} // namespace uzu
