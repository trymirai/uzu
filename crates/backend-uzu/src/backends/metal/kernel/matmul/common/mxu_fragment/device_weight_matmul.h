template <typename RightElement, class OutputFragment, class LeftFragment>
METAL_FUNC static void fragment_matmul_int8_device_weights(
    thread OutputFragment& output,
    thread LeftFragment& left,
    const device uchar* right_signed_codes,
    const int right_row_stride_bytes
) {
  static_assert(RELAXED, "device weight tensors require the relaxed MXU layout");
  static_assert(LeftFragment::COL_FRAGMENTS == 2, "device weight tensors expect K tiled as two fragments");
  static_assert(OutputFragment::COL_FRAGMENTS % 2 == 0, "device weight tensors require even N fragments");
  static_assert(LeftFragment::ROW_FRAGMENTS == OutputFragment::ROW_FRAGMENTS, "M tiles must match");

  constexpr ushort rows = OutputFragment::ROW_FRAGMENTS;
  constexpr ushort cols = OutputFragment::COL_FRAGMENTS;
  constexpr int tile_k = int(2 * FRAGMENT_COLS);
  constexpr int tile_n = int(2 * FRAGMENT_COLS);
  constexpr int element_bits = metal::is_same_v<RightElement, metal::int4b_format> ? 4 : 8;
  constexpr int elements_per_byte = 8 / element_bits;
  using RightTensor = tensor<device RightElement, extents<int, tile_k, tile_n>, tensor_inline>;
  using RightPointer =
      metal::conditional_t<metal::is_same_v<RightElement, metal::int4b_format>, device uchar*, device RightElement*>;

  constexpr auto descriptor =
      mpp::tensor_ops::matmul2d_descriptor(FRAGMENT_ROWS, tile_n, tile_k, false, true, RELAXED, MatmulMode::multiply);
  mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

  const array<int, 2> right_strides = {1, right_row_stride_bytes * elements_per_byte};

  METAL_PRAGMA_UNROLL
  for (ushort row = 0; row < rows; ++row) {
    METAL_PRAGMA_UNROLL
    for (ushort col = 0; col < cols; col += 2) {
      auto cooperative_left = matmul_op.template get_left_input_cooperative_tensor<int8_t, RightElement, int>();
      load_paired_vectors(cooperative_left, left.fragment_at(row, 0), left.fragment_at(row, 1));

      RightTensor right_tensor(
          // MPP rejects const-qualified tensor element types even though the
          // right operand is read-only during matmul.
          reinterpret_cast<RightPointer>(
              const_cast<device uchar*>(right_signed_codes + int(col * FRAGMENT_COLS) * right_row_stride_bytes)
          ),
          extents<int, tile_k, tile_n>{},
          right_strides
      );

      auto cooperative_output =
          matmul_op.template get_destination_cooperative_tensor<decltype(cooperative_left), RightTensor, int>();

      matmul_op.run(cooperative_left, right_tensor, cooperative_output);

      store_paired_vectors(cooperative_output, output.fragment_at(row, col), output.fragment_at(row, col + 1));
    }
  }
}
