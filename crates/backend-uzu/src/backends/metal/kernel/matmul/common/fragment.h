#pragma once

#include <metal_stdlib>

#include "../../common/integral_constant.h"
#include "../../common/thread_context.h"
#include "defines.h"

using namespace metal;

namespace uzu {
namespace matmul {

template <typename Ptr>
struct TileSource {
  Ptr base;
  int row_stride;
  int col_stride;
  short row_bound;
  short col_bound;
  bool ragged;

  METAL_FUNC TileSource advanced(int elements) const thread {
    TileSource out = *this;
    out.base += elements;
    return out;
  }

  METAL_FUNC TileSource bounded(short rows, short cols) const thread {
    TileSource out = *this;
    out.row_bound = rows;
    out.col_bound = cols;
    out.ragged = true;
    return out;
  }
};

template <typename Ptr>
METAL_FUNC TileSource<Ptr> tile_source(Ptr base, int row_stride, int col_stride = 1) {
  return TileSource<Ptr>{base, row_stride, col_stride, 0, 0, false};
}

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
      Ops::ELEMENTS_PER_THREAD == (Ops::FRAGMENT_ROWS * Ops::FRAGMENT_COLS) / METAL_SIMD_SIZE,
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

  METAL_FUNC static constexpr short2 get_position(const ushort simd_lane_id) {
    const short quad = simd_lane_id / 4;
    const short row = (quad & 4) + (simd_lane_id / 2) % 4;
    const short col = ((quad & 2) + simd_lane_id % 2) * Ops::THREAD_ELEMENT_COLS;
    return short2{col, row};
  }

  METAL_FUNC constexpr void clear() {
    METAL_PRAGMA_UNROLL
    for (ushort index = 0; index < NUM_FRAGS; ++index) {
      fragment_data[index] = ThreadVectorType(0);
    }
  }

  METAL_FUNC constexpr thread ThreadVectorType& fragment_at(const ushort row_index, const ushort col_index) {
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

  METAL_FUNC thread ElementType* elements() { return reinterpret_cast<thread ElementType*>(fragment_data); }

  template <class Fn>
  METAL_FUNC void for_each_element(const ushort simd_lane_id, Fn fn) thread {
    const short2 position = get_position(simd_lane_id);
    thread ElementType* data = elements();
    for_each_fragment([&](auto tile_row, auto tile_col) {
      const short row_base = position.y + short(tile_row.value) * FRAGMENT_ROWS;
      const short col_base = position.x + short(tile_col.value) * FRAGMENT_COLS;
      const ushort frag_base = (tile_row.value * TILE_COLS + tile_col.value) * Ops::ELEMENTS_PER_THREAD;
      METAL_PRAGMA_UNROLL
      for (ushort i = 0; i < Ops::THREAD_ELEMENT_ROWS; ++i) {
        METAL_PRAGMA_UNROLL
        for (ushort j = 0; j < Ops::THREAD_ELEMENT_COLS; ++j) {
          const short row = row_base + short(i) * Ops::THREAD_ELEMENT_ROW_STRIDE;
          const short col = col_base + short(j);
          fn(row, col, data[frag_base + i * Ops::THREAD_ELEMENT_COLS + j]);
        }
      }
    });
  }

  // Row lanes differ in bits {0,3} for both simdgroup and MXU layouts.
  template <typename Acc, class Fn>
  METAL_FUNC void row_reduce(thread Acc* out, const Acc identity, Fn op) thread {
    thread ElementType* data = elements();
    METAL_PRAGMA_UNROLL
    for (ushort tr = 0; tr < TILE_ROWS; ++tr) {
      METAL_PRAGMA_UNROLL
      for (ushort i = 0; i < Ops::THREAD_ELEMENT_ROWS; ++i) {
        Acc acc = identity;
        METAL_PRAGMA_UNROLL
        for (ushort tc = 0; tc < TILE_COLS; ++tc) {
          const ushort frag_base = (tr * TILE_COLS + tc) * Ops::ELEMENTS_PER_THREAD;
          METAL_PRAGMA_UNROLL
          for (ushort j = 0; j < Ops::THREAD_ELEMENT_COLS; ++j) {
            acc = op(acc, Acc(data[frag_base + i * Ops::THREAD_ELEMENT_COLS + j]));
          }
        }
        acc = op(acc, simd_shuffle_xor(acc, 1));
        acc = op(acc, simd_shuffle_xor(acc, 8));
        out[tr * Ops::THREAD_ELEMENT_ROWS + i] = acc;
      }
    }
  }

  template <typename Acc, class Fn>
  METAL_FUNC void row_bin_op(const thread Acc* row_vals, Fn fn) thread {
    thread ElementType* data = elements();
    METAL_PRAGMA_UNROLL
    for (ushort tr = 0; tr < TILE_ROWS; ++tr) {
      METAL_PRAGMA_UNROLL
      for (ushort i = 0; i < Ops::THREAD_ELEMENT_ROWS; ++i) {
        const Acc rv = row_vals[tr * Ops::THREAD_ELEMENT_ROWS + i];
        METAL_PRAGMA_UNROLL
        for (ushort tc = 0; tc < TILE_COLS; ++tc) {
          const ushort frag_base = (tr * TILE_COLS + tc) * Ops::ELEMENTS_PER_THREAD;
          METAL_PRAGMA_UNROLL
          for (ushort j = 0; j < Ops::THREAD_ELEMENT_COLS; ++j) {
            const ushort e = frag_base + i * Ops::THREAD_ELEMENT_COLS + j;
            data[e] = ElementType(fn(Acc(data[e]), rv));
          }
        }
      }
    }
  }

  template <
      class Ptr,
      class RowStride,
      class ColStride = Int<1>,
      class TileRowStride = Int<1>,
      class TileColStride = Int<1>>
  METAL_FUNC void load(
      const ushort simd_lane_id,
      Ptr source,
      RowStride row_stride,
      ColStride col_stride = {},
      TileRowStride tile_row_stride = {},
      TileColStride tile_col_stride = {}
  ) thread {
    transfer<LOAD, UNSAFE>(
        simd_lane_id,
        source,
        row_stride,
        col_stride,
        Int<0>{},
        Int<0>{},
        tile_row_stride,
        tile_col_stride
    );
  }

  template <class Ptr, class TileRowStride = Int<1>, class TileColStride = Int<1>>
  METAL_FUNC void load_safe(
      const ushort simd_lane_id,
      Ptr source,
      const int leading_dimension,
      const short2 tile_dimensions,
      TileRowStride tile_row_stride = {},
      TileColStride tile_col_stride = {}
  ) thread {
    transfer<LOAD, SAFE>(
        simd_lane_id,
        source,
        leading_dimension,
        Int<1>{},
        tile_dimensions.y,
        tile_dimensions.x,
        tile_row_stride,
        tile_col_stride
    );
  }

  template <class Ptr>
  METAL_FUNC void load_from(const ushort simd_lane_id, TileSource<Ptr> src) thread {
    if (src.ragged) {
      transfer<LOAD, SAFE>(
          simd_lane_id,
          src.base,
          src.row_stride,
          src.col_stride,
          src.row_bound,
          src.col_bound,
          Int<1>{},
          Int<1>{}
      );
    } else {
      transfer<LOAD, UNSAFE>(
          simd_lane_id,
          src.base,
          src.row_stride,
          src.col_stride,
          Int<0>{},
          Int<0>{},
          Int<1>{},
          Int<1>{}
      );
    }
  }

  template <
      class Ptr,
      class RowStride,
      class ColStride = Int<1>,
      class TileRowStride = Int<1>,
      class TileColStride = Int<1>>
  METAL_FUNC void store(
      const ushort simd_lane_id,
      Ptr destination,
      RowStride row_stride,
      ColStride col_stride = {},
      TileRowStride tile_row_stride = {},
      TileColStride tile_col_stride = {}
  ) thread {
    transfer<STORE, UNSAFE>(
        simd_lane_id,
        destination,
        row_stride,
        col_stride,
        Int<0>{},
        Int<0>{},
        tile_row_stride,
        tile_col_stride
    );
  }

  template <class Ptr, class TileRowStride = Int<1>, class TileColStride = Int<1>>
  METAL_FUNC void store_safe(
      const ushort simd_lane_id,
      Ptr destination,
      const int leading_dimension,
      const short2 tile_dimensions,
      TileRowStride tile_row_stride = {},
      TileColStride tile_col_stride = {}
  ) thread {
    transfer<STORE, SAFE>(
        simd_lane_id,
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
      const_for_loop<0, TILE_COLS, 1>([&](auto col_index) { fn(row_index, col_index); });
    });
  }

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
      const ushort simd_lane_id,
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

    const short2 position = get_position(simd_lane_id);
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
          const auto offset = col_stride_is_one ? (row * row_stride + col) : (row * row_stride + col * col_stride);

          if constexpr (IS_LOAD) {
            if constexpr (IS_SAFE) {
              const bool in_bounds = (row < local_row_limit) && (col < local_col_limit);
              frag[element_index] = in_bounds ? static_cast<T>(ptr[offset]) : T(0);
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

template <bool transpose_a = false, bool transpose_b = false, class OutputTile, class LeftTile, class RightTile>
METAL_FUNC void tile_matmul(thread OutputTile& output, thread LeftTile& left, thread RightTile& right) {
  static_assert(
      metal::is_same_v<typename OutputTile::FragmentOpsType, typename LeftTile::FragmentOpsType> &&
          metal::is_same_v<typename OutputTile::FragmentOpsType, typename RightTile::FragmentOpsType>,
      "tile_matmul requires output, left, and right tiles to use the same FragmentOps"
  );
  OutputTile::FragmentOpsType::template tile_matmul<transpose_a, transpose_b>(output, left, right);
}

} // namespace matmul
} // namespace uzu
