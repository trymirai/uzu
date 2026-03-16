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

///////////////////////////////////////////////////////////////////////////////
// SimdgroupFragment - thread's portion of simdgroup tile data
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    int GRID_ROWS_,
    int GRID_COLS_,
    class SimdgroupMultiplyAccumulateOps_ = SimdgroupMultiplyAccumulate<T, 8, 8>>
struct SimdgroupFragment {
  using SimdgroupMultiplyAccumulateOpsType = SimdgroupMultiplyAccumulateOps_;
  using ElementType = T;
  METAL_CONST int MULTIPLY_ACCUMULATE_ROWS = SimdgroupMultiplyAccumulateOpsType::ROWS;
  METAL_CONST int MULTIPLY_ACCUMULATE_COLS = SimdgroupMultiplyAccumulateOpsType::COLS;
  METAL_CONST int ELEMENTS_PER_MULTIPLY_ACCUMULATE = SimdgroupMultiplyAccumulateOpsType::THREAD_ELEMENT_COUNT;

  METAL_CONST int GRID_ROWS = GRID_ROWS_;
  METAL_CONST int GRID_COLS = GRID_COLS_;

  METAL_CONST int TOTAL_ROWS = GRID_ROWS * MULTIPLY_ACCUMULATE_ROWS;
  METAL_CONST int TOTAL_COLS = GRID_COLS * MULTIPLY_ACCUMULATE_COLS;

  METAL_CONST int MULTIPLY_ACCUMULATE_COUNT = GRID_ROWS * GRID_COLS;
  METAL_CONST int ELEMENTS_PER_FRAGMENT = MULTIPLY_ACCUMULATE_COUNT * ELEMENTS_PER_MULTIPLY_ACCUMULATE;

  typedef typename SimdgroupMultiplyAccumulateOpsType::SimdgroupMatrixType SimdgroupMatrixType;
  typedef typename SimdgroupMultiplyAccumulateOpsType::ThreadDataType ThreadDataType;

  ThreadDataType multiply_accumulate_data[MULTIPLY_ACCUMULATE_COUNT] = {ThreadDataType(0)};

  METAL_FUNC SimdgroupFragment() thread {}

  METAL_FUNC constexpr void clear() {
    METAL_PRAGMA_UNROLL
    for (short i = 0; i < MULTIPLY_ACCUMULATE_COUNT; ++i) {
      multiply_accumulate_data[i] = ThreadDataType(0);
    }
  }

  METAL_FUNC constexpr thread ThreadDataType& multiply_accumulate_at(const short i, const short j) {
    return multiply_accumulate_data[i * GRID_COLS + j];
  }

  METAL_FUNC constexpr const thread ThreadDataType& multiply_accumulate_at(
      const short i,
      const short j
  ) const {
    return multiply_accumulate_data[i * GRID_COLS + j];
  }

  METAL_FUNC thread ElementType* elements() {
    return reinterpret_cast<thread ElementType*>(multiply_accumulate_data);
  }

  METAL_FUNC const thread ElementType* elements() const {
    return reinterpret_cast<const thread ElementType*>(multiply_accumulate_data);
  }

  template <typename U, int SIMDGROUP_STRIDE_X, int SIMDGROUP_STRIDE_Y, int STRIDE_X, int STRIDE_Y>
  METAL_FUNC void load(const threadgroup U* src) {
    METAL_PRAGMA_UNROLL
    for (short i = 0; i < GRID_ROWS; ++i) {
      METAL_PRAGMA_UNROLL
      for (short j = 0; j < GRID_COLS; ++j) {
        SimdgroupMultiplyAccumulateOpsType::load(
            multiply_accumulate_at(i, j),
            &(
                src[(i * MULTIPLY_ACCUMULATE_ROWS) * SIMDGROUP_STRIDE_X * STRIDE_X +
                    (j * MULTIPLY_ACCUMULATE_COLS) * SIMDGROUP_STRIDE_Y * STRIDE_Y]
            ),
            STRIDE_X,
            STRIDE_Y
        );
      }
    }
  }

  template <typename U, int SIMDGROUP_STRIDE_X, int SIMDGROUP_STRIDE_Y>
  METAL_FUNC void store(device U* dst, const int leading_dimension) const {
    METAL_PRAGMA_UNROLL
    for (short i = 0; i < GRID_ROWS; ++i) {
      METAL_PRAGMA_UNROLL
      for (short j = 0; j < GRID_COLS; ++j) {
        SimdgroupMultiplyAccumulateOpsType::store(
            multiply_accumulate_at(i, j),
            &(dst[(i * MULTIPLY_ACCUMULATE_ROWS) * SIMDGROUP_STRIDE_X * leading_dimension + (j * MULTIPLY_ACCUMULATE_COLS) * SIMDGROUP_STRIDE_Y]),
            leading_dimension,
            1
        );
      }
    }
  }

  template <typename U, int SIMDGROUP_STRIDE_X, int SIMDGROUP_STRIDE_Y>
  METAL_FUNC void store_safe(
      device U* dst,
      const int leading_dimension,
      const short2 destination_tile_dimensions
  ) const {
    METAL_PRAGMA_UNROLL
    for (int i = 0; i < GRID_ROWS; ++i) {
      METAL_PRAGMA_UNROLL
      for (int j = 0; j < GRID_COLS; ++j) {
        SimdgroupMultiplyAccumulateOpsType::store_safe(
            multiply_accumulate_at(i, j),
            dst,
            leading_dimension,
            1,
            destination_tile_dimensions.y,
            destination_tile_dimensions.x,
            (i * MULTIPLY_ACCUMULATE_ROWS) * SIMDGROUP_STRIDE_X,
            (j * MULTIPLY_ACCUMULATE_COLS) * SIMDGROUP_STRIDE_Y
        );
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
// Tile multiply-accumulate
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U, int M, int N, int K>
METAL_FUNC void tile_multiply_accumulate(
    thread SimdgroupFragment<T, M, N>& D,
    thread SimdgroupFragment<U, M, K>& A,
    thread SimdgroupFragment<U, K, N>& B,
    thread SimdgroupFragment<T, M, N>& C
) {
  METAL_PRAGMA_UNROLL
  for (short m = 0; m < M; ++m) {
    METAL_PRAGMA_UNROLL
    for (short n = 0; n < N; ++n) {
      short column_serpentine = (m % 2) ? (N - 1 - n) : n;
      METAL_PRAGMA_UNROLL
      for (short k = 0; k < K; ++k) {
        SimdgroupFragment<T, M, N>::SimdgroupMultiplyAccumulateOpsType::multiply_accumulate(
            D.multiply_accumulate_at(m, column_serpentine),
            A.multiply_accumulate_at(m, k),
            B.multiply_accumulate_at(k, column_serpentine),
            C.multiply_accumulate_at(m, column_serpentine)
        );
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// ThreadgroupTile - manages the GEMM computation for a threadgroup
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    typename U,
    int BLOCK_ROWS,
    int BLOCK_COLS,
    int BLOCK_DEPTH,
    int SIMDGROUPS_PER_ROW,
    int SIMDGROUPS_PER_COLUMN,
    bool transpose_a,
    bool transpose_b,
    short THREADGROUP_LEADING_DIMENSION_A,
    short THREADGROUP_LEADING_DIMENSION_B,
    typename AccumType = float,
    typename Epilogue = TransformNone<U, AccumType>>
struct ThreadgroupTile {
  METAL_CONST short SIMDGROUP_BLOCK_SIZE = 8;
  using SimdgroupMultiplyAccumulateType = SimdgroupMultiplyAccumulate<AccumType, SIMDGROUP_BLOCK_SIZE, SIMDGROUP_BLOCK_SIZE>;

  METAL_CONST short TILE_ROW_STRIDE = SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_ROW;
  METAL_CONST short TILE_COL_STRIDE = SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_COLUMN;

  METAL_CONST short TILE_ROWS = BLOCK_ROWS / (SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_ROW);
  METAL_CONST short TILE_COLS = BLOCK_COLS / (SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_COLUMN);

  METAL_CONST short A_STRIDE_ROW = transpose_a ? 1 : THREADGROUP_LEADING_DIMENSION_A;
  METAL_CONST short A_STRIDE_INNER = transpose_a ? THREADGROUP_LEADING_DIMENSION_A : 1;

  METAL_CONST short B_STRIDE_INNER = transpose_b ? 1 : THREADGROUP_LEADING_DIMENSION_B;
  METAL_CONST short B_STRIDE_COL = transpose_b ? THREADGROUP_LEADING_DIMENSION_B : 1;

  METAL_CONST short TILE_STRIDE_A = SIMDGROUP_BLOCK_SIZE * A_STRIDE_INNER;
  METAL_CONST short TILE_STRIDE_B = SIMDGROUP_BLOCK_SIZE * B_STRIDE_INNER;

  SimdgroupFragment<AccumType, TILE_ROWS, 1, SimdgroupMultiplyAccumulateType> a_fragment;
  SimdgroupFragment<AccumType, 1, TILE_COLS, SimdgroupMultiplyAccumulateType> b_fragment;
  SimdgroupFragment<AccumType, TILE_ROWS, TILE_COLS, SimdgroupMultiplyAccumulateType> c_fragment;

  short simdgroup_row_offset;
  short simdgroup_col_offset;

  short a_shared_offset;
  short b_shared_offset;

  METAL_FUNC ThreadgroupTile(
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]
  ) {
    short tile_row_base = SIMDGROUP_BLOCK_SIZE * (simd_group_id / SIMDGROUPS_PER_COLUMN);
    short tile_col_base = SIMDGROUP_BLOCK_SIZE * (simd_group_id % SIMDGROUPS_PER_COLUMN);

    short2 simdgroup_coordinates = SimdgroupMultiplyAccumulateType::get_lane_coordinates(simd_lane_id);
    simdgroup_row_offset = simdgroup_coordinates.y;
    simdgroup_col_offset = simdgroup_coordinates.x;

    a_shared_offset = (tile_row_base + simdgroup_row_offset) * A_STRIDE_ROW + (simdgroup_col_offset) * A_STRIDE_INNER;
    b_shared_offset = (simdgroup_row_offset) * B_STRIDE_INNER + (tile_col_base + simdgroup_col_offset) * B_STRIDE_COL;

    simdgroup_row_offset += tile_row_base;
    simdgroup_col_offset += tile_col_base;
  }

  METAL_FUNC void multiply_accumulate(const threadgroup T* a_shared, const threadgroup T* b_shared) {
    a_shared += a_shared_offset;
    b_shared += b_shared_offset;

    METAL_PRAGMA_UNROLL
    for (short k_block_index = 0; k_block_index < BLOCK_DEPTH; k_block_index += SIMDGROUP_BLOCK_SIZE) {
      simdgroup_barrier(mem_flags::mem_none);

      a_fragment.template load<T, SIMDGROUPS_PER_ROW, 1, A_STRIDE_ROW, A_STRIDE_INNER>(a_shared);

      simdgroup_barrier(mem_flags::mem_none);

      b_fragment.template load<T, 1, SIMDGROUPS_PER_COLUMN, B_STRIDE_INNER, B_STRIDE_COL>(b_shared);

      simdgroup_barrier(mem_flags::mem_none);

      tile_multiply_accumulate(c_fragment, a_fragment, b_fragment, c_fragment);

      a_shared += TILE_STRIDE_A;
      b_shared += TILE_STRIDE_B;
    }
  }

  METAL_FUNC void store_result(device U* D, const int leading_dimension_d) {
    METAL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(c_fragment)::ELEMENTS_PER_FRAGMENT; i++) {
      c_fragment.elements()[i] = Epilogue::apply(c_fragment.elements()[i]);
    }

    D += simdgroup_row_offset * leading_dimension_d + simdgroup_col_offset;

    c_fragment.template store<U, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN>(D, leading_dimension_d);
  }

  METAL_FUNC void store_result_safe(
      device U* D,
      const int leading_dimension_d,
      short2 destination_tile_dimensions
  ) {
    METAL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(c_fragment)::ELEMENTS_PER_FRAGMENT; i++) {
      c_fragment.elements()[i] = Epilogue::apply(c_fragment.elements()[i]);
    }

    D += simdgroup_row_offset * leading_dimension_d + simdgroup_col_offset;
    destination_tile_dimensions -= short2(simdgroup_col_offset, simdgroup_row_offset);

    if (destination_tile_dimensions.x <= 0 || destination_tile_dimensions.y <= 0)
      return;

    c_fragment.template store_safe<U, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN>(D, leading_dimension_d, destination_tile_dimensions);
  }

  template <typename EpilogueOp>
  METAL_FUNC void apply_epilogue(
      const device U* C,
      const int leading_dimension_c,
      const int column_stride_c,
      thread const EpilogueOp& epilogue_operation
  ) {
    const device U* c_pointer = C + simdgroup_row_offset * leading_dimension_c + simdgroup_col_offset * column_stride_c;

    METAL_PRAGMA_UNROLL
    for (short i = 0; i < TILE_ROWS; i++) {
      METAL_PRAGMA_UNROLL
      for (short j = 0; j < TILE_COLS; j++) {
        thread auto& block_data = c_fragment.multiply_accumulate_at(i, j);
        METAL_PRAGMA_UNROLL
        for (short k = 0; k < decltype(c_fragment)::ELEMENTS_PER_MULTIPLY_ACCUMULATE; k++) {
          short row_offset = (i * SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_ROW);
          short col_offset = (j * SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_COLUMN);
          U c_value = c_pointer[row_offset * leading_dimension_c + col_offset * column_stride_c + k];
          block_data[k] = epilogue_operation.apply(block_data[k], static_cast<AccumType>(c_value));
        }
      }
    }
  }

  template <typename EpilogueOp>
  METAL_FUNC void apply_epilogue_safe(
      const device U* C,
      const int leading_dimension_c,
      const int column_stride_c,
      short2 tile_dimensions,
      thread const EpilogueOp& epilogue_operation
  ) {
    const device U* c_pointer = C + simdgroup_row_offset * leading_dimension_c + simdgroup_col_offset * column_stride_c;
    tile_dimensions -= short2(simdgroup_col_offset, simdgroup_row_offset);

    METAL_PRAGMA_UNROLL
    for (short i = 0; i < TILE_ROWS; i++) {
      METAL_PRAGMA_UNROLL
      for (short j = 0; j < TILE_COLS; j++) {
        thread auto& block_data = c_fragment.multiply_accumulate_at(i, j);
        short row_offset = (i * SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_ROW);
        short col_offset = (j * SIMDGROUP_BLOCK_SIZE * SIMDGROUPS_PER_COLUMN);
        METAL_PRAGMA_UNROLL
        for (short k = 0; k < decltype(c_fragment)::ELEMENTS_PER_MULTIPLY_ACCUMULATE; k++) {
          if (row_offset < tile_dimensions.y && col_offset + k < tile_dimensions.x) {
            U c_value = c_pointer[row_offset * leading_dimension_c + col_offset * column_stride_c + k];
            block_data[k] = epilogue_operation.apply(block_data[k], static_cast<AccumType>(c_value));
          }
        }
      }
    }
  }
};

} // namespace matmul
} // namespace uzu
