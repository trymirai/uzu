// Included inside MxuFragmentOps; not a standalone header.

template <
    MatmulMode MODE,
    bool transpose_a,
    bool transpose_b,
    class OutputFragment,
    class LeftFragment,
    class RightFragment>
METAL_FUNC static void fragment_matmul(
    thread OutputFragment& output,
    thread LeftFragment& left,
    thread RightFragment& right
) {
  constexpr ushort left_rows = transpose_a ? LeftFragment::COL_FRAGMENTS : LeftFragment::ROW_FRAGMENTS;
  constexpr ushort rows = OutputFragment::ROW_FRAGMENTS;
  static_assert(left_rows == rows, "fragment matmul: M dimensions do not match");

  constexpr ushort right_cols = transpose_b ? RightFragment::ROW_FRAGMENTS : RightFragment::COL_FRAGMENTS;
  constexpr ushort cols = OutputFragment::COL_FRAGMENTS;
  static_assert(right_cols == cols, "fragment matmul: N dimensions do not match");

  constexpr ushort left_depth = transpose_a ? LeftFragment::ROW_FRAGMENTS : LeftFragment::COL_FRAGMENTS;
  constexpr ushort depth = transpose_b ? RightFragment::COL_FRAGMENTS : RightFragment::ROW_FRAGMENTS;
  static_assert(left_depth == depth, "fragment matmul: K dimensions do not match");

  static_assert(
      (cols % 2 == 0) || (cols == 1 && rows % 2 == 0),
      "MXU fragment_mma requires even N, or N==1 with even M (MPP pairing)"
  );

  constexpr auto transpose_left = metal::bool_constant<transpose_a>{};
  constexpr auto transpose_right = metal::bool_constant<transpose_b>{};

  if constexpr (cols == 1 && rows % 2 == 0) {
    METAL_PRAGMA_UNROLL
    for (ushort row = 0; row < rows; row += 2) {
      METAL_PRAGMA_UNROLL
      for (ushort col = 0; col < cols; ++col) {
        if constexpr (MODE == MatmulMode::multiply) {
          matmul<
              MatmulMode::multiply,
              typename OutputFragment::ElementType,
              typename LeftFragment::ElementType,
              typename RightFragment::ElementType,
              transpose_a,
              transpose_b>(
              output.fragment_at(row, col),
              output.fragment_at(row + 1, col),
              left.fragment_at(row, 0, transpose_left),
              left.fragment_at(row + 1, 0, transpose_left),
              transpose_left,
              right.fragment_at(0, col, transpose_right),
              transpose_right
          );
        }
        METAL_PRAGMA_UNROLL
        for (ushort k = MODE == MatmulMode::multiply_accumulate ? 0 : 1; k < depth; ++k) {
          matmul<
              MatmulMode::multiply_accumulate,
              typename OutputFragment::ElementType,
              typename LeftFragment::ElementType,
              typename RightFragment::ElementType,
              transpose_a,
              transpose_b>(
              output.fragment_at(row, col),
              output.fragment_at(row + 1, col),
              left.fragment_at(row, k, transpose_left),
              left.fragment_at(row + 1, k, transpose_left),
              transpose_left,
              right.fragment_at(k, col, transpose_right),
              transpose_right
          );
        }
      }
    }
  } else if constexpr (cols % 2 == 0) {
    METAL_PRAGMA_UNROLL
    for (ushort row = 0; row < rows; ++row) {
      METAL_PRAGMA_UNROLL
      for (ushort col = 0; col < cols; col += 2) {
        if constexpr (MODE == MatmulMode::multiply) {
          matmul<
              MatmulMode::multiply,
              typename OutputFragment::ElementType,
              typename LeftFragment::ElementType,
              typename RightFragment::ElementType,
              transpose_a,
              transpose_b>(
              output.fragment_at(row, col),
              output.fragment_at(row, col + 1),
              left.fragment_at(row, 0, transpose_left),
              transpose_left,
              right.fragment_at(0, col, transpose_right),
              right.fragment_at(0, col + 1, transpose_right),
              transpose_right
          );
        }
        METAL_PRAGMA_UNROLL
        for (ushort k = MODE == MatmulMode::multiply_accumulate ? 0 : 1; k < depth; ++k) {
          matmul<
              MatmulMode::multiply_accumulate,
              typename OutputFragment::ElementType,
              typename LeftFragment::ElementType,
              typename RightFragment::ElementType,
              transpose_a,
              transpose_b>(
              output.fragment_at(row, col),
              output.fragment_at(row, col + 1),
              left.fragment_at(row, k, transpose_left),
              transpose_left,
              right.fragment_at(k, col, transpose_right),
              right.fragment_at(k, col + 1, transpose_right),
              transpose_right
          );
        }
      }
    }
  }
}

template <bool transpose_a, bool transpose_b, class OutputFragment, class LeftFragment, class RightFragment>
METAL_FUNC static void fragment_mma(
    thread OutputFragment& output,
    thread LeftFragment& left,
    thread RightFragment& right
) {
  fragment_matmul<MatmulMode::multiply_accumulate, transpose_a, transpose_b>(output, left, right);
}

template <bool transpose_a, bool transpose_b, class OutputFragment, class LeftFragment, class RightFragment>
METAL_FUNC static void fragment_mm(
    thread OutputFragment& output,
    thread LeftFragment& left,
    thread RightFragment& right
) {
  // MXU relaxed multiply is slightly faster than multiply_accumulate for pure matmul.
  fragment_matmul<MatmulMode::multiply, transpose_a, transpose_b>(output, left, right);
}
