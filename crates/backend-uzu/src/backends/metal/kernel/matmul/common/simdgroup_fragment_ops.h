#pragma once

#include "defines.h"
#include "loader.h"
#include "simdgroup_multiply_accumulate.h"

using namespace metal;

namespace uzu {
namespace matmul {

struct SimdgroupFragmentOps {
  UZU_CONST ushort FRAGMENT_ROWS = SIMDGROUP_MMA_ROWS;
  UZU_CONST ushort FRAGMENT_COLS = SIMDGROUP_MMA_COLS;
  UZU_CONST bool READ_TRANSPOSE_SWAPS_SOURCE_STRIDES = true;
  using BlockStorage = ThreadgroupBlockStorage;

  UZU_CONST ushort ELEMENTS_PER_THREAD = (FRAGMENT_ROWS * FRAGMENT_COLS) / METAL_SIMD_SIZE;
  UZU_CONST ushort THREAD_ELEMENT_ROWS = 1;
  UZU_CONST ushort THREAD_ELEMENT_COLS = 2;
  UZU_CONST ushort THREAD_ELEMENT_ROW_STRIDE = FRAGMENT_ROWS / THREAD_ELEMENT_ROWS;

  template <typename U>
  using ThreadVector = typename metal::vec<U, ELEMENTS_PER_THREAD>;

  METAL_FUNC static constexpr short2 get_position(ushort simd_lane_id) {
    const short quad = simd_lane_id / 4;
    const short row = (quad & 4) + (simd_lane_id / 2) % 4;
    const short col = ((quad & 2) + simd_lane_id % 2) * THREAD_ELEMENT_COLS;
    return short2{col, row};
  }

  METAL_FUNC static constexpr short2 get_element_offset(ushort element_index) {
    return short2{short(element_index), 0};
  }

  template <bool transpose_a, bool transpose_b, class OutputFragment, class LeftFragment, class RightFragment>
  METAL_FUNC static void fragment_mma(
      thread OutputFragment& output,
      thread LeftFragment& left,
      thread RightFragment& right
  ) {
    static_assert(!transpose_a && !transpose_b, "SimdgroupFragmentOps::fragment_mma: transpose not supported");
    constexpr ushort rows = OutputFragment::ROW_FRAGMENTS;
    constexpr ushort cols = OutputFragment::COL_FRAGMENTS;
    constexpr ushort depth = LeftFragment::COL_FRAGMENTS;
    static_assert(LeftFragment::ROW_FRAGMENTS == rows, "fragment matmul: M dimensions do not match");
    static_assert(RightFragment::COL_FRAGMENTS == cols, "fragment matmul: N dimensions do not match");
    static_assert(RightFragment::ROW_FRAGMENTS == depth, "fragment matmul: K dimensions do not match");
    using Sg = SimdgroupMMA<
        typename OutputFragment::ElementType,
        FRAGMENT_ROWS,
        FRAGMENT_COLS,
        typename LeftFragment::ElementType,
        typename RightFragment::ElementType>;

    METAL_PRAGMA_UNROLL
    for (ushort m = 0; m < rows; ++m) {
      METAL_PRAGMA_UNROLL
      for (ushort n = 0; n < cols; ++n) {
        const ushort col = (m % 2) ? (cols - 1 - n) : n;
        METAL_PRAGMA_UNROLL
        for (ushort k = 0; k < depth; ++k) {
          Sg::mma(
              output.fragment_at(m, col),
              left.fragment_at(m, k),
              right.fragment_at(k, col),
              output.fragment_at(m, col)
          );
        }
      }
    }
  }

  template <bool transpose_a, bool transpose_b, class OutputFragment, class LeftFragment, class RightFragment>
  METAL_FUNC static void fragment_mm(
      thread OutputFragment& output,
      thread LeftFragment& left,
      thread RightFragment& right
  ) {
    // simdgroup_multiply is slower here.
    output.clear();
    fragment_mma<transpose_a, transpose_b>(output, left, right);
  }
};

template <typename U, int SIMDGROUP_STRIDE_X, int SIMDGROUP_STRIDE_Y, int STRIDE_X, int STRIDE_Y, typename FragT>
METAL_FUNC void fragment_load(thread FragT& frag, const threadgroup U* source) {
  constexpr ushort fragment_rows = SimdgroupFragmentOps::FRAGMENT_ROWS;
  constexpr ushort fragment_cols = SimdgroupFragmentOps::FRAGMENT_COLS;
  using Sg = SimdgroupMMA<typename FragT::ElementType, fragment_rows, fragment_cols>;
  METAL_PRAGMA_UNROLL
  for (ushort i = 0; i < FragT::ROW_FRAGMENTS; ++i) {
    METAL_PRAGMA_UNROLL
    for (ushort j = 0; j < FragT::COL_FRAGMENTS; ++j) {
      Sg::load(
          frag.fragment_at(i, j),
          &(source
                [(i * fragment_rows) * SIMDGROUP_STRIDE_X * STRIDE_X +
                 (j * fragment_cols) * SIMDGROUP_STRIDE_Y * STRIDE_Y]),
          STRIDE_X,
          STRIDE_Y
      );
    }
  }
}

template <typename U, int SIMDGROUP_STRIDE_X, int SIMDGROUP_STRIDE_Y, typename FragT>
METAL_FUNC void fragment_store(thread FragT& frag, device U* destination, const int leading_dimension) {
  constexpr ushort fragment_rows = SimdgroupFragmentOps::FRAGMENT_ROWS;
  constexpr ushort fragment_cols = SimdgroupFragmentOps::FRAGMENT_COLS;
  using Sg = SimdgroupMMA<typename FragT::ElementType, fragment_rows, fragment_cols>;
  METAL_PRAGMA_UNROLL
  for (ushort i = 0; i < FragT::ROW_FRAGMENTS; ++i) {
    METAL_PRAGMA_UNROLL
    for (ushort j = 0; j < FragT::COL_FRAGMENTS; ++j) {
      Sg::store(
          frag.fragment_at(i, j),
          &(destination
                [(i * fragment_rows) * SIMDGROUP_STRIDE_X * leading_dimension +
                 (j * fragment_cols) * SIMDGROUP_STRIDE_Y]),
          leading_dimension,
          1
      );
    }
  }
}

template <typename U, int SIMDGROUP_STRIDE_X, int SIMDGROUP_STRIDE_Y, typename FragT>
METAL_FUNC void fragment_store_safe(
    thread FragT& frag,
    device U* destination,
    const int leading_dimension,
    const short2 destination_tile_dimensions
) {
  constexpr ushort fragment_rows = SimdgroupFragmentOps::FRAGMENT_ROWS;
  constexpr ushort fragment_cols = SimdgroupFragmentOps::FRAGMENT_COLS;
  using Sg = SimdgroupMMA<typename FragT::ElementType, fragment_rows, fragment_cols>;
  METAL_PRAGMA_UNROLL
  for (int i = 0; i < FragT::ROW_FRAGMENTS; ++i) {
    METAL_PRAGMA_UNROLL
    for (int j = 0; j < FragT::COL_FRAGMENTS; ++j) {
      Sg::store_safe(
          frag.fragment_at(i, j),
          destination,
          leading_dimension,
          1,
          destination_tile_dimensions.y,
          destination_tile_dimensions.x,
          (i * fragment_rows) * SIMDGROUP_STRIDE_X,
          (j * fragment_cols) * SIMDGROUP_STRIDE_Y
      );
    }
  }
}

} // namespace matmul
} // namespace uzu
