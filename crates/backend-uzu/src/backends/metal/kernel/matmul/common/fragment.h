#pragma once

#include <metal_stdlib>

#include "../../common/integral_constant.h"
#include "../../common/thread_context.h"
#include "defines.h"

using namespace metal;

namespace uzu {
namespace matmul {

template <typename T, int TILE_ROWS_, int TILE_COLS_, class Ops>
struct Fragment {
  using FragmentOpsType = Ops;
  using ElementType = T;
  using ThreadVectorType = typename Ops::template ThreadVector<T>;

  METAL_CONST ushort TILE_ROWS = ushort(TILE_ROWS_);
  METAL_CONST ushort TILE_COLS = ushort(TILE_COLS_);

  METAL_CONST ushort FRAGMENT_ROWS = ushort(Ops::FRAGMENT_ROWS);
  METAL_CONST ushort FRAGMENT_COLS = ushort(Ops::FRAGMENT_COLS);

  METAL_CONST ushort NUM_FRAGS = TILE_ROWS * TILE_COLS;
  METAL_CONST ushort ELEMENTS_PER_TILE = NUM_FRAGS * Ops::ELEMENTS_PER_THREAD;

  ThreadVectorType fragment_data[NUM_FRAGS];
  ThreadContext thread_context;

  METAL_FUNC Fragment(const thread ThreadContext& thread_context) thread
      : thread_context(thread_context) {}

  METAL_FUNC constexpr void clear() {
    METAL_PRAGMA_UNROLL
    for (ushort index = 0; index < NUM_FRAGS; ++index) {
      fragment_data[index] = ThreadVectorType(0);
    }
  }

  METAL_FUNC constexpr thread ThreadVectorType& fragment_at(
      const ushort row_index,
      const ushort col_index
  ) {
    return fragment_data[row_index * TILE_COLS + col_index];
  }

  template <bool transpose>
  METAL_FUNC constexpr thread ThreadVectorType& fragment_at(
      const ushort row_index,
      const ushort col_index,
      metal::bool_constant<transpose>
  ) {
    if constexpr (transpose) {
      return fragment_at(col_index, row_index);
    } else {
      return fragment_at(row_index, col_index);
    }
  }

  METAL_FUNC thread ElementType* elements() {
    return reinterpret_cast<thread ElementType*>(fragment_data);
  }

  template <
      class Ptr,
      class RowStride,
      class ColStride,
      class TileRowStride = Int<1>,
      class TileColStride = Int<1>>
  METAL_FUNC void load(
      Ptr source,
      RowStride row_stride,
      ColStride col_stride,
      TileRowStride tile_row_stride = {},
      TileColStride tile_col_stride = {}
  ) thread {
    for_each_fragment([&](auto row_index, auto col_index) {
      Ops::load(
          fragment_at(row_index.value, col_index.value),
          source,
          row_stride,
          col_stride,
          row_index.value * FRAGMENT_ROWS * tile_row_stride,
          col_index.value * FRAGMENT_COLS * tile_col_stride,
          thread_context
      );
    });
  }

  template <class Ptr>
  METAL_FUNC void load(Ptr source, const int leading_dimension) thread {
    load(source, leading_dimension, Int<1>{});
  }

  template <
      class Ptr,
      class RowStride,
      class ColStride,
      class RowLimit,
      class ColLimit,
      class TileRowStride = Int<1>,
      class TileColStride = Int<1>>
  METAL_FUNC void load_safe(
      Ptr source,
      RowStride row_stride,
      ColStride col_stride,
      RowLimit row_limit,
      ColLimit col_limit,
      TileRowStride tile_row_stride = {},
      TileColStride tile_col_stride = {}
  ) thread {
    for_each_fragment([&](auto row_index, auto col_index) {
      Ops::load_safe(
          fragment_at(row_index.value, col_index.value),
          source,
          row_stride,
          col_stride,
          row_limit,
          col_limit,
          row_index.value * FRAGMENT_ROWS * tile_row_stride,
          col_index.value * FRAGMENT_COLS * tile_col_stride,
          thread_context
      );
    });
  }

  template <class Ptr>
  METAL_FUNC void load_safe(
      Ptr source,
      const int leading_dimension,
      const short2 tile_dimensions
  ) thread {
    load_safe(
        source,
        leading_dimension,
        Int<1>{},
        tile_dimensions.y,
        tile_dimensions.x
    );
  }

  template <
      class Ptr,
      class RowStride,
      class ColStride,
      class TileRowStride = Int<1>,
      class TileColStride = Int<1>>
  METAL_FUNC void store(
      Ptr destination,
      RowStride row_stride,
      ColStride col_stride,
      TileRowStride tile_row_stride = {},
      TileColStride tile_col_stride = {}
  ) thread {
    for_each_fragment([&](auto row_index, auto col_index) {
      Ops::store(
          fragment_at(row_index.value, col_index.value),
          destination,
          row_stride,
          col_stride,
          row_index.value * FRAGMENT_ROWS * tile_row_stride,
          col_index.value * FRAGMENT_COLS * tile_col_stride,
          thread_context
      );
    });
  }

  template <class Ptr>
  METAL_FUNC void store(Ptr destination, const int leading_dimension) thread {
    store(destination, leading_dimension, Int<1>{});
  }

  template <
      class Ptr,
      class RowStride,
      class ColStride,
      class RowLimit,
      class ColLimit,
      class TileRowStride = Int<1>,
      class TileColStride = Int<1>>
  METAL_FUNC void store_safe(
      Ptr destination,
      RowStride row_stride,
      ColStride col_stride,
      RowLimit row_limit,
      ColLimit col_limit,
      TileRowStride tile_row_stride = {},
      TileColStride tile_col_stride = {}
  ) thread {
    for_each_fragment([&](auto row_index, auto col_index) {
      Ops::store_safe(
          fragment_at(row_index.value, col_index.value),
          destination,
          row_stride,
          col_stride,
          row_limit,
          col_limit,
          row_index.value * FRAGMENT_ROWS * tile_row_stride,
          col_index.value * FRAGMENT_COLS * tile_col_stride,
          thread_context
      );
    });
  }

  template <class Ptr>
  METAL_FUNC void store_safe(
      Ptr destination,
      const int leading_dimension,
      const short2 tile_dimensions
  ) thread {
    store_safe(
        destination,
        leading_dimension,
        Int<1>{},
        tile_dimensions.y,
        tile_dimensions.x
    );
  }

 private:
  template <class Fn>
  METAL_FUNC void for_each_fragment(Fn fn) thread {
    const_for_loop<0, TILE_ROWS, 1>([&](auto row_index) {
      const_for_loop<0, TILE_COLS, 1>([&](auto col_index) {
        fn(row_index, col_index);
      });
    });
  }
};

} // namespace matmul
} // namespace uzu
