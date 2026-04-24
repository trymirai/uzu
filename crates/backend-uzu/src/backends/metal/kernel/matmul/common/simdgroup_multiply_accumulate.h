#pragma once

#include "defines.h"

using namespace metal;

namespace uzu {
namespace matmul {

///////////////////////////////////////////////////////////////////////////////
// SimdgroupMultiplyAccumulate - stateless traits/ops for 8x8 simdgroup_matrix
///////////////////////////////////////////////////////////////////////////////

template <typename T, int ROWS_, int COLS_>
struct SimdgroupMultiplyAccumulate {
  static_assert(
      ROWS_ == 8,
      "Only 8 x 8 fragment matrices are currently supported"
  );
  static_assert(
      COLS_ == 8,
      "Only 8 x 8 fragment matrices are currently supported"
  );
};

template <typename T>
struct SimdgroupMultiplyAccumulate<T, 8, 8> {
  METAL_CONST int ROWS = 8;
  METAL_CONST int COLS = 8;
  METAL_CONST int THREAD_ELEMENT_COUNT = (ROWS * COLS) / METAL_SIMD_SIZE;
  METAL_CONST int THREAD_ELEMENT_ROWS = 1;
  METAL_CONST int THREAD_ELEMENT_COLS = 2;

  static_assert(
      THREAD_ELEMENT_ROWS * THREAD_ELEMENT_COLS == THREAD_ELEMENT_COUNT,
      "SimdgroupMultiplyAccumulate shape is not consistent with element count"
  );

  typedef metal::simdgroup_matrix<T, ROWS, COLS> SimdgroupMatrixType;
  typedef metal::vec<T, THREAD_ELEMENT_COUNT> ThreadDataType;

  METAL_FUNC static constexpr short2 get_lane_coordinates(
      ushort simd_lane_id [[thread_index_in_simdgroup]]
  ) {
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
        destination[i * THREAD_ELEMENT_COLS + j] =
            static_cast<T>(source[i * row_stride + j * col_stride]);
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
      typename ColStride>
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
        destination[i * row_stride + j * col_stride] =
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
          destination
              [(row_offset + i) * row_stride + (col_offset + j) * col_stride] =
                  static_cast<U>(source[i * THREAD_ELEMENT_COLS + j]);
        }
      }
    }
  }

  METAL_FUNC static constexpr void multiply_accumulate(
      thread ThreadDataType& D,
      thread ThreadDataType& A,
      thread ThreadDataType& B,
      thread ThreadDataType& C
  ) {
    SimdgroupMatrixType D_mat;
    SimdgroupMatrixType A_mat;
    SimdgroupMatrixType B_mat;
    SimdgroupMatrixType C_mat;

    reinterpret_cast<thread ThreadDataType&>(A_mat.thread_elements()) = A;
    reinterpret_cast<thread ThreadDataType&>(B_mat.thread_elements()) = B;
    reinterpret_cast<thread ThreadDataType&>(C_mat.thread_elements()) = C;

    multiply_accumulate(D_mat, A_mat, B_mat, C_mat);

    D = reinterpret_cast<thread ThreadDataType&>(D_mat.thread_elements());
  }

  METAL_FUNC static constexpr void multiply_accumulate(
      thread SimdgroupMatrixType& D,
      thread SimdgroupMatrixType& A,
      thread SimdgroupMatrixType& B,
      thread SimdgroupMatrixType& C
  ) {
    simdgroup_multiply_accumulate(D, A, B, C);
  }
};

} // namespace matmul
} // namespace uzu
