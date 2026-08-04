template <typename Format>
METAL_FUNC static auto make_tensor_view(
    const typename Format::DevicePointer tile_base,
    const int right_row_stride_bytes
) {
  constexpr int tile_k = int(2 * FRAGMENT_COLS);
  constexpr int tile_n = int(2 * FRAGMENT_COLS);
  using Element = typename Format::TensorElement;
  using RightTensor = tensor<device Element, extents<int, tile_k, tile_n>, tensor_inline>;

  const array<int, 2> right_strides = {1, right_row_stride_bytes * Format::ELEMENTS_PER_BYTE};

  return RightTensor(
      const_cast<typename Format::MutableDevicePointer>(tile_base),
      extents<int, tile_k, tile_n>{},
      right_strides
  );
}

template <bool transpose_left, bool transpose_right, class OutputFragment, class LeftFragment, typename Format>
METAL_FUNC static void fragment_mm(
    thread OutputFragment& output,
    thread LeftFragment& left,
    const DeviceTensorOperand<Format> right
) {
  static_assert(!transpose_left && transpose_right, "device-tensor leaf: only left x right^T is implemented");
  static_assert(
      metal::is_same_v<typename LeftFragment::ElementType, typename Format::UnpackedElement>,
      "device-tensor leaf: integer operand signedness must match"
  );
  static_assert(RELAXED, "device-tensor operands require the relaxed MXU layout");
  static_assert(LeftFragment::COL_FRAGMENTS == 2, "device-tensor operands expect K tiled as two fragments");
  static_assert(OutputFragment::COL_FRAGMENTS % 2 == 0, "device-tensor operands require even N fragments");
  static_assert(LeftFragment::ROW_FRAGMENTS == OutputFragment::ROW_FRAGMENTS, "M tiles must match");

  constexpr ushort rows = OutputFragment::ROW_FRAGMENTS;
  constexpr ushort cols = OutputFragment::COL_FRAGMENTS;
  constexpr int tile_k = int(2 * FRAGMENT_COLS);
  constexpr int tile_n = int(2 * FRAGMENT_COLS);
  constexpr auto descriptor =
      mpp::tensor_ops::matmul2d_descriptor(FRAGMENT_ROWS, tile_n, tile_k, false, true, RELAXED, MatmulMode::multiply);
  mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

  METAL_PRAGMA_UNROLL
  for (ushort row = 0; row < rows; ++row) {
    METAL_PRAGMA_UNROLL
    for (ushort col = 0; col < cols; col += 2) {
      auto cooperative_left = matmul_op.template get_left_input_cooperative_tensor<
          typename Format::UnpackedElement,
          typename Format::TensorElement,
          int>();
      load_paired_vectors(cooperative_left, left.fragment_at(row, 0), left.fragment_at(row, 1));

      auto right_tensor = make_tensor_view<Format>(
          right.base + int(col * FRAGMENT_COLS) * right.row_stride_bytes,
          right.row_stride_bytes
      );

      auto cooperative_output =
          matmul_op
              .template get_destination_cooperative_tensor<decltype(cooperative_left), decltype(right_tensor), int>();

      matmul_op.run(cooperative_left, right_tensor, cooperative_output);

      store_paired_vectors(cooperative_output, output.fragment_at(row, col), output.fragment_at(row, col + 1));
    }
  }
}
