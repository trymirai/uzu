#pragma once

#include "defines.h"
#include "simdgroup_multiply_accumulate.h"

using namespace metal;

namespace uzu {
namespace matmul {

struct SimdgroupFragmentOps {
  METAL_CONST ushort FRAGMENT_ROWS = 8;
  METAL_CONST ushort FRAGMENT_COLS = 8;

  METAL_CONST ushort ELEMENTS_PER_THREAD = (FRAGMENT_ROWS * FRAGMENT_COLS) / METAL_SIMD_SIZE;
  METAL_CONST ushort THREAD_ELEMENT_ROWS = 1;
  METAL_CONST ushort THREAD_ELEMENT_COLS = 2;
  METAL_CONST ushort THREAD_ELEMENT_ROW_STRIDE = 8;

  template <typename U>
  using ThreadVector = typename metal::vec<U, ELEMENTS_PER_THREAD>;

  // Serpentine order matches the chain kernel's register reuse.
  template <bool transpose_a, bool transpose_b, class OutputFragment, class LeftFragment, class RightFragment>
  METAL_FUNC static void fragment_matmul(
      thread OutputFragment& output,
      thread LeftFragment& left,
      thread RightFragment& right
  ) {
    static_assert(!transpose_a && !transpose_b, "SimdgroupFragmentOps::fragment_matmul: transpose not supported");
    constexpr ushort rows = OutputFragment::ROW_FRAGMENTS;
    constexpr ushort cols = OutputFragment::COL_FRAGMENTS;
    constexpr ushort depth = LeftFragment::COL_FRAGMENTS;
    static_assert(LeftFragment::ROW_FRAGMENTS == rows, "fragment matmul: M dimensions do not match");
    static_assert(RightFragment::COL_FRAGMENTS == cols, "fragment matmul: N dimensions do not match");
    static_assert(RightFragment::ROW_FRAGMENTS == depth, "fragment matmul: K dimensions do not match");
    using Sg = SimdgroupMultiplyAccumulate<typename OutputFragment::ElementType, FRAGMENT_ROWS, FRAGMENT_COLS>;

    METAL_PRAGMA_UNROLL
    for (ushort m = 0; m < rows; ++m) {
      METAL_PRAGMA_UNROLL
      for (ushort n = 0; n < cols; ++n) {
        const ushort col = (m % 2) ? (cols - 1 - n) : n;
        METAL_PRAGMA_UNROLL
        for (ushort k = 0; k < depth; ++k) {
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

template <typename U, int SIMDGROUP_STRIDE_X, int SIMDGROUP_STRIDE_Y, int STRIDE_X, int STRIDE_Y, typename FragT>
METAL_FUNC void fragment_load(thread FragT& frag, const threadgroup U* source) {
  using Sg = SimdgroupMultiplyAccumulate<typename FragT::ElementType, 8, 8>;
  METAL_PRAGMA_UNROLL
  for (ushort i = 0; i < FragT::ROW_FRAGMENTS; ++i) {
    METAL_PRAGMA_UNROLL
    for (ushort j = 0; j < FragT::COL_FRAGMENTS; ++j) {
      Sg::load(
          frag.fragment_at(i, j),
          &(source[(i * 8) * SIMDGROUP_STRIDE_X * STRIDE_X + (j * 8) * SIMDGROUP_STRIDE_Y * STRIDE_Y]),
          STRIDE_X,
          STRIDE_Y
      );
    }
  }
}

template <typename U, int SIMDGROUP_STRIDE_X, int SIMDGROUP_STRIDE_Y, typename FragT>
METAL_FUNC void fragment_store(thread FragT& frag, device U* destination, const int leading_dimension) {
  using Sg = SimdgroupMultiplyAccumulate<typename FragT::ElementType, 8, 8>;
  METAL_PRAGMA_UNROLL
  for (ushort i = 0; i < FragT::ROW_FRAGMENTS; ++i) {
    METAL_PRAGMA_UNROLL
    for (ushort j = 0; j < FragT::COL_FRAGMENTS; ++j) {
      Sg::store(
          frag.fragment_at(i, j),
          &(destination[(i * 8) * SIMDGROUP_STRIDE_X * leading_dimension + (j * 8) * SIMDGROUP_STRIDE_Y]),
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
  using Sg = SimdgroupMultiplyAccumulate<typename FragT::ElementType, 8, 8>;
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
          (i * 8) * SIMDGROUP_STRIDE_X,
          (j * 8) * SIMDGROUP_STRIDE_Y
      );
    }
  }
}

} // namespace matmul
} // namespace uzu
