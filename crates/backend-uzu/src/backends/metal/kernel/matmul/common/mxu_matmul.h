#pragma once

#include "mxu_fragment_layout.h"

namespace uzu {
namespace matmul {

template <
    typename T,
    ushort TILE_ROWS_,
    ushort TILE_COLS_,
    class Fragment = MxuFragment>
struct MxuTile {
  using FragmentType = Fragment;
  using ElementType = T;

  METAL_CONST ushort TILE_ROWS = TILE_ROWS_;
  METAL_CONST ushort TILE_COLS = TILE_COLS_;

  METAL_CONST ushort ROWS = TILE_ROWS * FragmentType::FRAGMENT_ROWS;
  METAL_CONST ushort COLS = TILE_COLS * FragmentType::FRAGMENT_COLS;

  METAL_CONST ushort NUM_FRAGS = TILE_ROWS * TILE_COLS;
  METAL_CONST ushort ELEMENTS_PER_TILE =
      NUM_FRAGS * FragmentType::ELEMENTS_PER_FRAG;

  METAL_CONST ushort ROWS_PER_THREAD = TILE_ROWS * FragmentType::ELEMENT_ROWS;
  METAL_CONST ushort COLS_PER_THREAD = TILE_COLS * FragmentType::ELEMENT_COLS;

  typedef typename FragmentType::template FragmentVector<T> FragmentVectorType;

  FragmentVectorType fragment_data[NUM_FRAGS];

  METAL_FUNC MxuTile() thread {}

  METAL_FUNC constexpr void clear() {
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < NUM_FRAGS; ++i) {
      fragment_data[i] = FragmentVectorType(0);
    }
  }

  METAL_FUNC constexpr thread FragmentVectorType& fragment_at(
      const ushort i,
      const ushort j
  ) {
    return fragment_data[i * TILE_COLS + j];
  }

  METAL_FUNC constexpr const thread FragmentVectorType& fragment_at(
      const ushort i,
      const ushort j
  ) const {
    return fragment_data[i * TILE_COLS + j];
  }

  template <int i, int j>
  METAL_FUNC constexpr thread FragmentVectorType& fragment_at() {
    return fragment_data[i * TILE_COLS + j];
  }

  template <int i, int j>
  METAL_FUNC constexpr const thread FragmentVectorType& fragment_at() const {
    return fragment_data[i * TILE_COLS + j];
  }

  template <bool transpose>
  METAL_FUNC constexpr thread FragmentVectorType& fragment_at(
      const ushort i,
      const ushort j,
      metal::bool_constant<transpose>
  ) {
    if constexpr (transpose) {
      return fragment_at(j, i);
    } else {
      return fragment_at(i, j);
    }
  }

  template <bool transpose>
  METAL_FUNC constexpr const thread FragmentVectorType& fragment_at(
      const ushort i,
      const ushort j,
      metal::bool_constant<transpose>
  ) const {
    if constexpr (transpose) {
      return fragment_at(j, i);
    } else {
      return fragment_at(i, j);
    }
  }

  METAL_FUNC thread ElementType* elems() {
    return reinterpret_cast<thread ElementType*>(fragment_data);
  }

  METAL_FUNC const thread ElementType* elems() const {
    return reinterpret_cast<const thread ElementType*>(fragment_data);
  }

  template <typename SourcePointerType>
  METAL_FUNC void load(SourcePointerType source, const int leading_dimension) {
    const_for_loop<0, TILE_ROWS, 1>([&](auto idx_row) {
      const_for_loop<0, TILE_COLS, 1>([&](auto idx_col) {
        FragmentType::load(
            fragment_at<idx_row.value, idx_col.value>(),
            source,
            leading_dimension,
            Int<1>{},
            idx_row * Int<FragmentType::FRAGMENT_ROWS>{},
            idx_col * Int<FragmentType::FRAGMENT_COLS>{}
        );
      });
    });
  }

  template <typename U>
  METAL_FUNC void store(
      device U* destination,
      const int leading_dimension
  ) const {
    const_for_loop<0, TILE_ROWS, 1>([&](auto idx_row) {
      const_for_loop<0, TILE_COLS, 1>([&](auto idx_col) {
        FragmentType::store(
            fragment_at<idx_row.value, idx_col.value>(),
            destination,
            leading_dimension,
            Int<1>{},
            idx_row * Int<FragmentType::FRAGMENT_ROWS>{},
            idx_col * Int<FragmentType::FRAGMENT_COLS>{}
        );
      });
    });
  }

  template <typename SourcePointerType>
  METAL_FUNC void load_safe(
      SourcePointerType source,
      const int leading_dimension,
      const short2 tile_dimensions
  ) {
    const_for_loop<0, TILE_ROWS, 1>([&](auto idx_row) {
      const_for_loop<0, TILE_COLS, 1>([&](auto idx_col) {
        FragmentType::load_safe(
            fragment_at<idx_row.value, idx_col.value>(),
            source,
            leading_dimension,
            Int<1>{},
            tile_dimensions.y,
            tile_dimensions.x,
            idx_row * Int<FragmentType::FRAGMENT_ROWS>{},
            idx_col * Int<FragmentType::FRAGMENT_COLS>{}
        );
      });
    });
  }

  template <typename U>
  METAL_FUNC void store_safe(
      device U* destination,
      const int leading_dimension,
      const short2 tile_dimensions
  ) const {
    const_for_loop<0, TILE_ROWS, 1>([&](auto idx_row) {
      const_for_loop<0, TILE_COLS, 1>([&](auto idx_col) {
        FragmentType::store_safe(
            fragment_at<idx_row.value, idx_col.value>(),
            destination,
            leading_dimension,
            Int<1>{},
            tile_dimensions.y,
            tile_dimensions.x,
            idx_row * Int<FragmentType::FRAGMENT_ROWS>{},
            idx_col * Int<FragmentType::FRAGMENT_COLS>{}
        );
      });
    });
  }
};

template <
    class OutputTile,
    class LeftTile,
    class RightTile,
    bool transpose_a,
    bool transpose_b>
METAL_FUNC void tile_matmad(
    thread OutputTile& output,
    thread LeftTile& left,
    metal::bool_constant<transpose_a>,
    thread RightTile& right,
    metal::bool_constant<transpose_b>
) {
  constexpr ushort left_tile_m =
      transpose_a ? LeftTile::TILE_COLS : LeftTile::TILE_ROWS;
  constexpr ushort tile_m = OutputTile::TILE_ROWS;
  static_assert(
      left_tile_m == tile_m,
      "tile matmul: M dimensions do not match"
  );

  constexpr ushort right_tile_n =
      transpose_b ? RightTile::TILE_ROWS : RightTile::TILE_COLS;
  constexpr ushort tile_n = OutputTile::TILE_COLS;
  static_assert(
      right_tile_n == tile_n,
      "tile matmul: N dimensions do not match"
  );

  constexpr ushort left_tile_k =
      transpose_a ? LeftTile::TILE_ROWS : LeftTile::TILE_COLS;
  constexpr ushort tile_k =
      transpose_b ? RightTile::TILE_COLS : RightTile::TILE_ROWS;
  static_assert(
      left_tile_k == tile_k,
      "tile matmul: K dimensions do not match"
  );

  constexpr auto transpose_left = metal::bool_constant<transpose_a>{};
  constexpr auto transpose_right = metal::bool_constant<transpose_b>{};

  if constexpr (tile_n == 1 && tile_m % 2 == 0) {
    METAL_PRAGMA_UNROLL
    for (ushort row = 0; row < tile_m; row += 2) {
      METAL_PRAGMA_UNROLL
      for (ushort col = 0; col < tile_n; ++col) {
        METAL_PRAGMA_UNROLL
        for (ushort k = 0; k < tile_k; ++k) {
          OutputTile::FragmentType::mma(
              output.fragment_at(row, col),
              output.fragment_at(row + 1, col),
              left.fragment_at(row, k, transpose_left),
              left.fragment_at(row + 1, k, transpose_left),
              metal::bool_constant<transpose_a>{},
              right.fragment_at(k, col, transpose_right),
              metal::bool_constant<transpose_b>{}
          );
        }
      }
    }
  } else if constexpr (tile_n % 2 == 0) {
    METAL_PRAGMA_UNROLL
    for (ushort row = 0; row < tile_m; ++row) {
      METAL_PRAGMA_UNROLL
      for (ushort col = 0; col < tile_n; col += 2) {
        METAL_PRAGMA_UNROLL
        for (ushort k = 0; k < tile_k; ++k) {
          OutputTile::FragmentType::mma(
              output.fragment_at(row, col),
              output.fragment_at(row, col + 1),
              left.fragment_at(row, k, transpose_left),
              metal::bool_constant<transpose_a>{},
              right.fragment_at(k, col, transpose_right),
              right.fragment_at(k, col + 1, transpose_right),
              metal::bool_constant<transpose_b>{}
          );
        }
      }
    }
  }
}

} // namespace matmul
} // namespace uzu
