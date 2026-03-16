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
  METAL_CONST int THREAD_ELEMENT_COUNT = (ROWS * COLS) / 32;
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

  template <typename SrcPtrType, typename StrideX, typename StrideY>
  METAL_FUNC static constexpr void load(
      thread ThreadDataType& dst,
      SrcPtrType src,
      StrideX stride_x,
      StrideY stride_y
  ) {
    METAL_PRAGMA_UNROLL
    for (short i = 0; i < THREAD_ELEMENT_ROWS; i++) {
      METAL_PRAGMA_UNROLL
      for (short j = 0; j < THREAD_ELEMENT_COLS; j++) {
        dst[i * THREAD_ELEMENT_COLS + j] = static_cast<T>(src[i * stride_x + j * stride_y]);
      }
    }
  }

  template <
      typename SrcPtrType,
      typename StrideX,
      typename StrideY,
      typename LimitX,
      typename LimitY,
      typename OffsetX,
      typename OffsetY>
  METAL_FUNC static constexpr void load_safe(
      thread ThreadDataType& dst,
      SrcPtrType src,
      StrideX stride_x,
      StrideY stride_y,
      LimitX limit_x,
      LimitY limit_y,
      OffsetX offset_x = 0,
      OffsetY offset_y = 0
  ) {
    METAL_PRAGMA_UNROLL
    for (short i = 0; i < THREAD_ELEMENT_ROWS; i++) {
      METAL_PRAGMA_UNROLL
      for (short j = 0; j < THREAD_ELEMENT_COLS; j++) {
        if ((offset_x + i) < limit_x && (offset_y + j) < limit_y) {
          dst[i * THREAD_ELEMENT_COLS + j] =
              static_cast<T>(src[(offset_x + i) * stride_x + (offset_y + j) * stride_y]);
        } else {
          dst[i * THREAD_ELEMENT_COLS + j] = T(0);
        }
      }
    }
  }

  template <typename DstPtrType, typename StrideX, typename StrideY>
  METAL_FUNC static constexpr void store(
      const thread ThreadDataType& src,
      DstPtrType dst,
      StrideX stride_x,
      StrideY stride_y
  ) {
    using U = PointerElementType<DstPtrType>;

    METAL_PRAGMA_UNROLL
    for (short i = 0; i < THREAD_ELEMENT_ROWS; i++) {
      METAL_PRAGMA_UNROLL
      for (short j = 0; j < THREAD_ELEMENT_COLS; j++) {
        dst[i * stride_x + j * stride_y] = static_cast<U>(src[i * THREAD_ELEMENT_COLS + j]);
      }
    }
  }

  template <
      typename DstPtrType,
      typename StrideX,
      typename StrideY,
      typename LimitX,
      typename LimitY,
      typename OffsetX,
      typename OffsetY>
  METAL_FUNC static constexpr void store_safe(
      const thread ThreadDataType& src,
      DstPtrType dst,
      StrideX stride_x,
      StrideY stride_y,
      LimitX limit_x,
      LimitY limit_y,
      OffsetX offset_x = 0,
      OffsetY offset_y = 0
  ) {
    using U = PointerElementType<DstPtrType>;

    METAL_PRAGMA_UNROLL
    for (short i = 0; i < THREAD_ELEMENT_ROWS; i++) {
      METAL_PRAGMA_UNROLL
      for (short j = 0; j < THREAD_ELEMENT_COLS; j++) {
        if ((offset_x + i) < limit_x && (offset_y + j) < limit_y) {
          dst[(offset_x + i) * stride_x + (offset_y + j) * stride_y] =
              static_cast<U>(src[i * THREAD_ELEMENT_COLS + j]);
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
