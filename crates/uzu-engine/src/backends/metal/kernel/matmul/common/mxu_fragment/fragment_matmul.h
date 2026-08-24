// Fragment storage is paired into the 16x32 cooperative tile used here.
template <
    MatmulMode OUTPUT_MODE,
    typename OutputType,
    typename LeftType,
    typename RightType,
    bool transpose_left,
    bool transpose_right,
    typename MarshalInputs>
METAL_FUNC static void matmul(
    thread ThreadVector<OutputType>& output_0,
    thread ThreadVector<OutputType>& output_1,
    MarshalInputs marshal_inputs
) {
  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      FRAGMENT_ROWS,
      2 * FRAGMENT_COLS,
      FRAGMENT_COLS,
      transpose_left,
      transpose_right,
      RELAXED,
      OUTPUT_MODE
  );

  mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

  auto cooperative_left = matmul_op.template get_left_input_cooperative_tensor<LeftType, RightType, OutputType>();
  auto cooperative_right = matmul_op.template get_right_input_cooperative_tensor<LeftType, RightType, OutputType>();
  auto cooperative_output = matmul_op.template get_destination_cooperative_tensor<
      decltype(cooperative_left),
      decltype(cooperative_right),
      OutputType>();

  marshal_inputs(cooperative_left, cooperative_right);

  if constexpr (OUTPUT_MODE == MatmulMode::multiply_accumulate) {
    load_paired_vectors(cooperative_output, output_0, output_1);
  }

  matmul_op.run(cooperative_left, cooperative_right, cooperative_output);

  store_paired_vectors(cooperative_output, output_0, output_1);
}

template <
    MatmulMode OUTPUT_MODE,
    bool transpose_left,
    bool transpose_right,
    class OutputFragment,
    class LeftFragment,
    class RightFragment>
METAL_FUNC static void fragment_matmul(
    thread OutputFragment& output,
    thread LeftFragment& left,
    thread RightFragment& right
) {
  constexpr ushort left_rows = transpose_left ? LeftFragment::COL_FRAGMENTS : LeftFragment::ROW_FRAGMENTS;
  constexpr ushort rows = OutputFragment::ROW_FRAGMENTS;
  static_assert(left_rows == rows, "fragment matmul: M dimensions do not match");

  constexpr ushort right_cols = transpose_right ? RightFragment::ROW_FRAGMENTS : RightFragment::COL_FRAGMENTS;
  constexpr ushort cols = OutputFragment::COL_FRAGMENTS;
  static_assert(right_cols == cols, "fragment matmul: N dimensions do not match");

  constexpr ushort left_depth = transpose_left ? LeftFragment::ROW_FRAGMENTS : LeftFragment::COL_FRAGMENTS;
  constexpr ushort depth = transpose_right ? RightFragment::COL_FRAGMENTS : RightFragment::ROW_FRAGMENTS;
  static_assert(left_depth == depth, "fragment matmul: K dimensions do not match");

  static_assert(
      (cols % 2 == 0) || (cols == 1 && rows % 2 == 0),
      "MXU fragment_mma requires even N, or N==1 with even M (MPP pairing)"
  );

  constexpr auto left_transpose = metal::bool_constant<transpose_left>{};
  constexpr auto right_transpose = metal::bool_constant<transpose_right>{};
  constexpr bool pair_output_rows = (cols == 1 && rows % 2 == 0);

  auto matmul_paired_outputs = [&](ushort row, ushort col, ushort depth_index, auto output_mode) {
    constexpr MatmulMode mode = decltype(output_mode)::value;
    if constexpr (pair_output_rows) {
      static_assert(RELAXED, "strict MXU row-pairing is not implemented");
      const thread auto& left_row_0 = left.fragment_at(row, depth_index, left_transpose);
      const thread auto& left_row_1 = left.fragment_at(row + 1, depth_index, left_transpose);
      const thread auto& right_operand = right.fragment_at(depth_index, col, right_transpose);
      matmul<
          mode,
          typename OutputFragment::ElementType,
          typename LeftFragment::ElementType,
          typename RightFragment::ElementType,
          transpose_left,
          transpose_right>(
          output.fragment_at(row, col),
          output.fragment_at(row + 1, col),
          [&](thread auto& cooperative_left, thread auto& cooperative_right) {
            load_paired_vectors(cooperative_left, left_row_0, left_row_1);
            METAL_PRAGMA_UNROLL
            for (ushort i = 0; i < ELEMENTS_PER_THREAD; i++) {
              cooperative_right[i] = right_operand[i];
            }
          }
      );
    } else {
      const thread auto& left_operand = left.fragment_at(row, depth_index, left_transpose);
      const thread auto& right_col_0 = right.fragment_at(depth_index, col, right_transpose);
      const thread auto& right_col_1 = right.fragment_at(depth_index, col + 1, right_transpose);
      matmul<
          mode,
          typename OutputFragment::ElementType,
          typename LeftFragment::ElementType,
          typename RightFragment::ElementType,
          transpose_left,
          transpose_right>(
          output.fragment_at(row, col),
          output.fragment_at(row, col + 1),
          [&](thread auto& cooperative_left, thread auto& cooperative_right) {
            METAL_PRAGMA_UNROLL
            for (ushort i = 0; i < ELEMENTS_PER_THREAD; i++) {
              cooperative_left[i] = left_operand[i];
            }
            load_paired_vectors(cooperative_right, right_col_0, right_col_1);
          }
      );
    }
  };

  constexpr ushort output_row_step = pair_output_rows ? 2 : 1;
  constexpr ushort output_col_count = pair_output_rows ? 1 : cols;
  constexpr ushort output_col_step = pair_output_rows ? 1 : 2;

  METAL_PRAGMA_UNROLL
  for (ushort row = 0; row < rows; row += output_row_step) {
    METAL_PRAGMA_UNROLL
    for (ushort col = 0; col < output_col_count; col += output_col_step) {
      if constexpr (OUTPUT_MODE == MatmulMode::multiply) {
        matmul_paired_outputs(row, col, 0, uzu::integral_constant<MatmulMode, MatmulMode::multiply>{});
      }
      METAL_PRAGMA_UNROLL
      for (ushort depth_index = OUTPUT_MODE == MatmulMode::multiply_accumulate ? 0 : 1; depth_index < depth;
           ++depth_index) {
        matmul_paired_outputs(
            row,
            col,
            depth_index,
            uzu::integral_constant<MatmulMode, MatmulMode::multiply_accumulate>{}
        );
      }
    }
  }
}

template <bool transpose_left, bool transpose_right, class OutputFragment, class LeftFragment, class RightFragment>
METAL_FUNC static void fragment_mma(
    thread OutputFragment& output,
    thread LeftFragment& left,
    thread RightFragment& right
) {
  fragment_matmul<MatmulMode::multiply_accumulate, transpose_left, transpose_right>(output, left, right);
}

template <bool transpose_left, bool transpose_right, class OutputFragment, class LeftFragment, class RightFragment>
METAL_FUNC static void fragment_mm(
    thread OutputFragment& output,
    thread LeftFragment& left,
    thread RightFragment& right
) {
  // MXU relaxed multiply is slightly faster than multiply_accumulate for pure matmul.
  fragment_matmul<MatmulMode::multiply, transpose_left, transpose_right>(output, left, right);
}
