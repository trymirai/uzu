// MPP has no valid 16x16x16 op; fragment_mma pairs fragments into 16x32.
template <
    MatmulMode MODE,
    typename CType,
    typename AType,
    typename BType,
    bool transpose_a,
    bool transpose_b,
    typename MarshalInputs>
METAL_FUNC static void mma_impl(
    thread ThreadVector<CType>& output_0,
    thread ThreadVector<CType>& output_1,
    MarshalInputs marshal_inputs
) {
  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      FRAGMENT_ROWS,
      2 * FRAGMENT_COLS,
      FRAGMENT_COLS,
      transpose_a,
      transpose_b,
      RELAXED,
      MODE
  );

  mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

  auto cooperative_left = matmul_op.template get_left_input_cooperative_tensor<AType, BType, CType>();
  auto cooperative_right = matmul_op.template get_right_input_cooperative_tensor<AType, BType, CType>();
  auto cooperative_output = matmul_op.template get_destination_cooperative_tensor<
      decltype(cooperative_left),
      decltype(cooperative_right),
      CType>();

  marshal_inputs(cooperative_left, cooperative_right);

  if constexpr (MODE == MatmulMode::multiply_accumulate) {
    load_paired_vectors(cooperative_output, output_0, output_1);
  }

  matmul_op.run(cooperative_left, cooperative_right, cooperative_output);

  store_paired_vectors(cooperative_output, output_0, output_1);
}

template <
    MatmulMode MODE,
    typename CType,
    typename AType,
    typename BType,
    bool transpose_a = false,
    bool transpose_b = false>
METAL_FUNC static void matmul(
    thread ThreadVector<CType>& output_col_0,
    thread ThreadVector<CType>& output_col_1,
    const thread ThreadVector<AType>& left,
    metal::bool_constant<transpose_a>,
    const thread ThreadVector<BType>& right_col_0,
    const thread ThreadVector<BType>& right_col_1,
    metal::bool_constant<transpose_b>
) {
  mma_impl<MODE, CType, AType, BType, transpose_a, transpose_b>(
      output_col_0,
      output_col_1,
      [&](thread auto& cooperative_left, thread auto& cooperative_right) {
        METAL_PRAGMA_UNROLL
        for (ushort i = 0; i < ELEMENTS_PER_THREAD; i++) {
          cooperative_left[i] = left[i];
        }
        load_paired_vectors(cooperative_right, right_col_0, right_col_1);
      }
  );
}

template <
    MatmulMode MODE,
    typename CType,
    typename AType,
    typename BType,
    bool transpose_a = false,
    bool transpose_b = false>
METAL_FUNC static void matmul(
    thread ThreadVector<CType>& output_row_0,
    thread ThreadVector<CType>& output_row_1,
    const thread ThreadVector<AType>& left_row_0,
    const thread ThreadVector<AType>& left_row_1,
    metal::bool_constant<transpose_a>,
    const thread ThreadVector<BType>& right,
    metal::bool_constant<transpose_b>
) {
  static_assert(RELAXED, "strict MXU row-pairing is not implemented");
  mma_impl<MODE, CType, AType, BType, transpose_a, transpose_b>(
      output_row_0,
      output_row_1,
      [&](thread auto& cooperative_left, thread auto& cooperative_right) {
        load_paired_vectors(cooperative_left, left_row_0, left_row_1);
        METAL_PRAGMA_UNROLL
        for (ushort i = 0; i < ELEMENTS_PER_THREAD; i++) {
          cooperative_right[i] = right[i];
        }
      }
  );
}
