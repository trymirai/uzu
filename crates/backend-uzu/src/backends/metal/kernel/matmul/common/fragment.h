#pragma once

#include <metal_stdlib>

#include "../../common/integral_constant.h"
#include "../../common/thread_context.h"
#include "defines.h"

using namespace metal;

namespace uzu {
namespace matmul {

///////////////////////////////////////////////////////////////////////////////
// Fragment - a thread-private tile of values arranged as TILE_ROWS x TILE_COLS
// sub-tiles. Ops provides the sub-tile shape constants and the MMA primitive;
// Fragment itself owns lane positioning and device<->register transfer.
///////////////////////////////////////////////////////////////////////////////

template <typename T, ushort TILE_ROWS_, ushort TILE_COLS_, class Ops>
struct Fragment {
  using FragmentOpsType = Ops;
  using ElementType = T;
  using ThreadVectorType = typename Ops::template ThreadVector<T>;

  static_assert(
      Ops::FRAGMENT_ROWS > 0 && Ops::FRAGMENT_COLS > 0,
      "Ops must expose positive FRAGMENT_ROWS/FRAGMENT_COLS"
  );
  static_assert(
      Ops::ELEMENTS_PER_THREAD ==
          (Ops::FRAGMENT_ROWS * Ops::FRAGMENT_COLS) / METAL_SIMD_SIZE,
      "Ops::ELEMENTS_PER_THREAD must equal "
      "(FRAGMENT_ROWS * FRAGMENT_COLS) / METAL_SIMD_SIZE"
  );
  static_assert(
      sizeof(ThreadVectorType) == Ops::ELEMENTS_PER_THREAD * sizeof(T),
      "Ops::ThreadVector<T> must be tightly packed "
      "(elements() relies on it)"
  );

  METAL_CONST ushort TILE_ROWS = TILE_ROWS_;
  METAL_CONST ushort TILE_COLS = TILE_COLS_;

  METAL_CONST ushort FRAGMENT_ROWS = ushort(Ops::FRAGMENT_ROWS);
  METAL_CONST ushort FRAGMENT_COLS = ushort(Ops::FRAGMENT_COLS);

  METAL_CONST ushort NUM_FRAGS = TILE_ROWS * TILE_COLS;
  METAL_CONST ushort ELEMENTS_PER_TILE = NUM_FRAGS * Ops::ELEMENTS_PER_THREAD;

  ThreadVectorType fragment_data[NUM_FRAGS];
  ThreadContext thread_context;

  METAL_FUNC Fragment(const thread ThreadContext& thread_context) thread
      : thread_context(thread_context) {}

  // Lane origin (row, col) within a single FRAGMENT_ROWS x FRAGMENT_COLS
  // sub-tile. The mapping depends only on Ops::THREAD_ELEMENT_COLS, so it is
  // identical for every Fragment sharing that Ops type.
  METAL_FUNC static constexpr short2 get_position(
      const thread ThreadContext& thread_context
  ) {
    const ushort simdgroup_index = ushort(thread_context.simdgroup_index);
    const short quad = simdgroup_index / 4;
    const short row = (quad & 4) + (simdgroup_index / 2) % 4;
    const short col =
        ((quad & 2) + simdgroup_index % 2) * Ops::THREAD_ELEMENT_COLS;
    return short2{col, row};
  }

  METAL_FUNC short2 get_position() const thread {
    return get_position(thread_context);
  }

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

  // Unsafe load: copy a (TILE_ROWS * FRAGMENT_ROWS) x (TILE_COLS * FRAGMENT_COLS)
  // block from device memory into fragment registers. ColStride defaults to the
  // compile-time constant 1 so that the row-major fast path triggers when the
  // caller omits it. Tile strides control the gap between sub-tiles.
  template <
      class Ptr,
      class RowStride,
      class ColStride = Int<1>,
      class TileRowStride = Int<1>,
      class TileColStride = Int<1>>
  METAL_FUNC void load(
      Ptr source,
      RowStride row_stride,
      ColStride col_stride = {},
      TileRowStride tile_row_stride = {},
      TileColStride tile_col_stride = {}
  ) thread {
    transfer<LOAD, UNSAFE>(
        source,
        row_stride,
        col_stride,
        Int<0>{},
        Int<0>{},
        tile_row_stride,
        tile_col_stride
    );
  }

  // Safe load: col_stride is implicitly 1; out-of-bounds elements become T(0).
  template <class Ptr, class TileRowStride = Int<1>, class TileColStride = Int<1>>
  METAL_FUNC void load_safe(
      Ptr source,
      const int leading_dimension,
      const short2 tile_dimensions,
      TileRowStride tile_row_stride = {},
      TileColStride tile_col_stride = {}
  ) thread {
    transfer<LOAD, SAFE>(
        source,
        leading_dimension,
        Int<1>{},
        tile_dimensions.y,
        tile_dimensions.x,
        tile_row_stride,
        tile_col_stride
    );
  }

  // Unsafe store: mirror of load. Same defaults apply.
  template <
      class Ptr,
      class RowStride,
      class ColStride = Int<1>,
      class TileRowStride = Int<1>,
      class TileColStride = Int<1>>
  METAL_FUNC void store(
      Ptr destination,
      RowStride row_stride,
      ColStride col_stride = {},
      TileRowStride tile_row_stride = {},
      TileColStride tile_col_stride = {}
  ) thread {
    transfer<STORE, UNSAFE>(
        destination,
        row_stride,
        col_stride,
        Int<0>{},
        Int<0>{},
        tile_row_stride,
        tile_col_stride
    );
  }

  // Safe store: col_stride is implicitly 1; out-of-bounds elements are skipped.
  template <class Ptr, class TileRowStride = Int<1>, class TileColStride = Int<1>>
  METAL_FUNC void store_safe(
      Ptr destination,
      const int leading_dimension,
      const short2 tile_dimensions,
      TileRowStride tile_row_stride = {},
      TileColStride tile_col_stride = {}
  ) thread {
    transfer<STORE, SAFE>(
        destination,
        leading_dimension,
        Int<1>{},
        tile_dimensions.y,
        tile_dimensions.x,
        tile_row_stride,
        tile_col_stride
    );
  }

private:
  METAL_CONST bool LOAD = true;
  METAL_CONST bool STORE = false;
  METAL_CONST bool SAFE = true;
  METAL_CONST bool UNSAFE = false;

  template <class Fn>
  METAL_FUNC void for_each_fragment(Fn fn) thread {
    const_for_loop<0, TILE_ROWS, 1>([&](auto row_index) {
      const_for_loop<0, TILE_COLS, 1>([&](auto col_index) {
        fn(row_index, col_index);
      });
    });
  }

  // Unified memory transfer between device memory and fragment registers.
  //
  // IS_LOAD  - true to read into fragment_data, false to write back.
  // IS_SAFE  - true to bounds-check each element against (row_limit, col_limit);
  //            on load, out-of-bounds elements become T(0); on store, skipped.
  //
  // When ColStride is a compile-time 1, the column-stride multiplication is
  // elided to help the compiler collapse strength-reduced addressing.
  template <
      bool IS_LOAD,
      bool IS_SAFE,
      class Ptr,
      class RowStride,
      class ColStride,
      class RowLimit,
      class ColLimit,
      class TileRowStride,
      class TileColStride>
  METAL_FUNC void transfer(
      Ptr ptr,
      RowStride row_stride,
      ColStride col_stride,
      RowLimit row_limit,
      ColLimit col_limit,
      TileRowStride tile_row_stride,
      TileColStride tile_col_stride
  ) thread {
    using U = PointerElementType<Ptr>;
    constexpr bool col_stride_is_one = metal::is_same_v<ColStride, Int<1>>;

    const short2 position = get_position();
    ptr += position.y * row_stride + position.x * col_stride;
    const auto local_row_limit = row_limit - position.y;
    const auto local_col_limit = col_limit - position.x;

    for_each_fragment([&](auto tile_row, auto tile_col) {
      thread auto& frag = fragment_at(tile_row.value, tile_col.value);
      const auto row_base = tile_row.value * FRAGMENT_ROWS * tile_row_stride;
      const auto col_base = tile_col.value * FRAGMENT_COLS * tile_col_stride;

      METAL_PRAGMA_UNROLL
      for (ushort i = 0; i < Ops::THREAD_ELEMENT_ROWS; i++) {
        const auto row = row_base + i * Ops::THREAD_ELEMENT_ROW_STRIDE;
        METAL_PRAGMA_UNROLL
        for (ushort j = 0; j < Ops::THREAD_ELEMENT_COLS; j++) {
          const ushort element_index = i * Ops::THREAD_ELEMENT_COLS + j;
          const auto col = col_base + j;
          const auto offset = col_stride_is_one
              ? (row * row_stride + col)
              : (row * row_stride + col * col_stride);

          if constexpr (IS_LOAD) {
            if constexpr (IS_SAFE) {
              const bool in_bounds =
                  (row < local_row_limit) && (col < local_col_limit);
              frag[element_index] =
                  in_bounds ? static_cast<T>(ptr[offset]) : T(0);
            } else {
              frag[element_index] = static_cast<T>(ptr[offset]);
            }
          } else {
            if constexpr (IS_SAFE) {
              if ((row < local_row_limit) && (col < local_col_limit)) {
                ptr[offset] = static_cast<U>(frag[element_index]);
              }
            } else {
              ptr[offset] = static_cast<U>(frag[element_index]);
            }
          }
        }
      }
    });
  }
};

} // namespace matmul
} // namespace uzu
