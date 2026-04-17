#pragma once

#include "simdgroup_multiply_accumulate.h"

using namespace metal;

namespace uzu {
namespace matmul {

///////////////////////////////////////////////////////////////////////////////
// SimdgroupFragment - thread's portion of simdgroup tile data
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    int GRID_ROWS_,
    int GRID_COLS_,
    class SimdgroupMultiplyAccumulateOps_ =
        SimdgroupMultiplyAccumulate<T, 8, 8>>
struct SimdgroupFragment {
  using SimdgroupMultiplyAccumulateOpsType = SimdgroupMultiplyAccumulateOps_;
  using ElementType = T;
  METAL_CONST int MULTIPLY_ACCUMULATE_ROWS =
      SimdgroupMultiplyAccumulateOpsType::ROWS;
  METAL_CONST int MULTIPLY_ACCUMULATE_COLS =
      SimdgroupMultiplyAccumulateOpsType::COLS;
  METAL_CONST int ELEMENTS_PER_MULTIPLY_ACCUMULATE =
      SimdgroupMultiplyAccumulateOpsType::THREAD_ELEMENT_COUNT;

  METAL_CONST int GRID_ROWS = GRID_ROWS_;
  METAL_CONST int GRID_COLS = GRID_COLS_;

  METAL_CONST int TOTAL_ROWS = GRID_ROWS * MULTIPLY_ACCUMULATE_ROWS;
  METAL_CONST int TOTAL_COLS = GRID_COLS * MULTIPLY_ACCUMULATE_COLS;

  METAL_CONST int MULTIPLY_ACCUMULATE_COUNT = GRID_ROWS * GRID_COLS;
  METAL_CONST int ELEMENTS_PER_FRAGMENT =
      MULTIPLY_ACCUMULATE_COUNT * ELEMENTS_PER_MULTIPLY_ACCUMULATE;

  typedef typename SimdgroupMultiplyAccumulateOpsType::SimdgroupMatrixType
      SimdgroupMatrixType;
  typedef typename SimdgroupMultiplyAccumulateOpsType::ThreadDataType
      ThreadDataType;

  ThreadDataType multiply_accumulate_data[MULTIPLY_ACCUMULATE_COUNT] = {
      ThreadDataType(0)
  };

  METAL_FUNC SimdgroupFragment() thread {}

  METAL_FUNC constexpr void clear() {
    METAL_PRAGMA_UNROLL
    for (short i = 0; i < MULTIPLY_ACCUMULATE_COUNT; ++i) {
      multiply_accumulate_data[i] = ThreadDataType(0);
    }
  }

  METAL_FUNC constexpr thread ThreadDataType& multiply_accumulate_at(
      const short i,
      const short j
  ) {
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
    return reinterpret_cast<const thread ElementType*>(
        multiply_accumulate_data
    );
  }

  template <
      typename U,
      int SIMDGROUP_STRIDE_X,
      int SIMDGROUP_STRIDE_Y,
      int STRIDE_X,
      int STRIDE_Y>
  METAL_FUNC void load(const threadgroup U* src) {
    METAL_PRAGMA_UNROLL
    for (short i = 0; i < GRID_ROWS; ++i) {
      METAL_PRAGMA_UNROLL
      for (short j = 0; j < GRID_COLS; ++j) {
        SimdgroupMultiplyAccumulateOpsType::load(
            multiply_accumulate_at(i, j),
            &(
                src[(i * MULTIPLY_ACCUMULATE_ROWS) * SIMDGROUP_STRIDE_X *
                        STRIDE_X +
                    (j * MULTIPLY_ACCUMULATE_COLS) * SIMDGROUP_STRIDE_Y *
                        STRIDE_Y]
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
            &(
                dst[(i * MULTIPLY_ACCUMULATE_ROWS) * SIMDGROUP_STRIDE_X *
                        leading_dimension +
                    (j * MULTIPLY_ACCUMULATE_COLS) * SIMDGROUP_STRIDE_Y]
            ),
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
        SimdgroupFragment<T, M, N>::SimdgroupMultiplyAccumulateOpsType::
            multiply_accumulate(
                D.multiply_accumulate_at(m, column_serpentine),
                A.multiply_accumulate_at(m, k),
                B.multiply_accumulate_at(k, column_serpentine),
                C.multiply_accumulate_at(m, column_serpentine)
            );
      }
    }
  }
}

} // namespace matmul
} // namespace uzu
