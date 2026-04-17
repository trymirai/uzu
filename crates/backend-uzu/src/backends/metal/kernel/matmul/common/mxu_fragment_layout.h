#pragma once

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

#include "../../common/defines.h"
#include "../../common/integral_constant.h"
using namespace uzu;

#include "matmul_support.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

namespace uzu {
namespace matmul {

struct MxuFragment {
  METAL_CONST ushort FRAGMENT_ROWS = 16;
  METAL_CONST ushort FRAGMENT_COLS = 16;

  METAL_CONST ushort ELEMENTS_PER_FRAG =
      (FRAGMENT_ROWS * FRAGMENT_COLS) / METAL_SIMD_SIZE;

  METAL_CONST ushort ELEMENT_ROWS = 2;
  METAL_CONST ushort ELEMENT_COLS = 4;

  METAL_CONST ushort ELEMENT_ROWS_JUMP = 8;

  static_assert(
      ELEMENT_ROWS * ELEMENT_COLS == ELEMENTS_PER_FRAG,
      "MxuFragment shape is not consistent with element count"
  );

  template <typename U>
  using FragmentVector = typename metal::vec<U, ELEMENTS_PER_FRAG>;

  METAL_FUNC static short2 get_lane_coordinate() {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short quad_id = simd_lane_id >> 2;
    const short lane_row = ((quad_id & 4) | ((simd_lane_id >> 1) & 3));
    const short lane_col = ((quad_id & 2) | (simd_lane_id & 1)) * 4;
    return short2{lane_col, lane_row};
  }

  METAL_FUNC static short2 get_lane_coordinate(ushort index) {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short quad_id = simd_lane_id >> 2;
    const short lane_row =
        ((quad_id & 4) | ((simd_lane_id >> 1) & 3)) + (index >> 2) * 8;
    const short lane_col = ((quad_id & 2) | (simd_lane_id & 1)) * 4 + index % 4;
    return short2{lane_col, lane_row};
  }

  template <
      typename T,
      typename SourcePointerType,
      typename RowStride,
      typename ColStride,
      typename RowOffset = Int<0>,
      typename ColOffset = Int<0>>
  METAL_FUNC static constexpr void load(
      thread FragmentVector<T>& destination,
      SourcePointerType source,
      RowStride row_stride,
      ColStride col_stride,
      RowOffset row_offset = {},
      ColOffset col_offset = {}
  ) {
    const short2 lane_coord = get_lane_coordinate();
    source += lane_coord.y * row_stride + lane_coord.x * col_stride;

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENT_ROWS; i++) {
      const auto row = row_offset + i * ELEMENT_ROWS_JUMP;
      const auto col = col_offset;

      if constexpr (metal::is_same_v<ColStride, Int<1>>) {
        METAL_PRAGMA_UNROLL
        for (ushort j = 0; j < ELEMENT_COLS; j++) {
          destination[i * ELEMENT_COLS + j] =
              static_cast<T>(source[row * row_stride + col + j]);
        }
      } else {
        METAL_PRAGMA_UNROLL
        for (ushort j = 0; j < ELEMENT_COLS; j++) {
          destination[i * ELEMENT_COLS + j] =
              static_cast<T>(source[row * row_stride + (col + j) * col_stride]);
        }
      }
    }
  }

  template <
      typename T,
      typename SourcePointerType,
      typename RowStride,
      typename ColStride,
      typename RowLimit,
      typename RowOffset = Int<0>,
      typename ColOffset = Int<0>>
  METAL_FUNC static constexpr void load_rows(
      thread FragmentVector<T>& destination,
      SourcePointerType source,
      RowStride row_stride,
      ColStride col_stride,
      RowLimit row_limit,
      RowOffset row_offset = {},
      ColOffset col_offset = {}
  ) {
    const short2 lane_coord = get_lane_coordinate();
    source += lane_coord.y * row_stride + lane_coord.x * col_stride;
    auto local_row_limit = row_limit - lane_coord.y;

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENT_ROWS; i++) {
      const auto row = row_offset + i * ELEMENT_ROWS_JUMP;
      const auto col = col_offset;

      if (row < local_row_limit) {
        if constexpr (metal::is_same_v<ColStride, Int<1>>) {
          METAL_PRAGMA_UNROLL
          for (ushort j = 0; j < ELEMENT_COLS; j++) {
            destination[i * ELEMENT_COLS + j] =
                static_cast<T>(source[row * row_stride + (col + j)]);
          }
        } else {
          METAL_PRAGMA_UNROLL
          for (ushort j = 0; j < ELEMENT_COLS; j++) {
            destination[i * ELEMENT_COLS + j] = static_cast<T>(
                source[row * row_stride + (col + j) * col_stride]
            );
          }
        }
      } else {
        METAL_PRAGMA_UNROLL
        for (ushort j = 0; j < ELEMENT_COLS; j++) {
          destination[i * ELEMENT_COLS + j] = T(0);
        }
      }
    }
  }

  template <
      typename T,
      typename SourcePointerType,
      typename RowStride,
      typename ColStride,
      typename RowLimit,
      typename ColLimit,
      typename RowOffset = Int<0>,
      typename ColOffset = Int<0>>
  METAL_FUNC static constexpr void load_safe(
      thread FragmentVector<T>& destination,
      SourcePointerType source,
      RowStride row_stride,
      ColStride col_stride,
      RowLimit row_limit,
      ColLimit col_limit,
      RowOffset row_offset = {},
      ColOffset col_offset = {}
  ) {
    const short2 lane_coord = get_lane_coordinate();
    source += lane_coord.y * row_stride + lane_coord.x * col_stride;
    auto local_row_limit = row_limit - lane_coord.y;
    auto local_col_limit = col_limit - lane_coord.x;

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENT_ROWS; i++) {
      const auto row = row_offset + i * ELEMENT_ROWS_JUMP;
      const auto col = col_offset;
      METAL_PRAGMA_UNROLL
      for (ushort j = 0; j < ELEMENT_COLS; j++) {
        if ((row < local_row_limit) && ((col + j) < local_col_limit)) {
          destination[i * ELEMENT_COLS + j] =
              static_cast<T>(source[row * row_stride + (col + j) * col_stride]);
        } else {
          destination[i * ELEMENT_COLS + j] = T(0);
        }
      }
    }
  }

  template <
      typename T,
      typename DestinationPointerType,
      typename RowStride,
      typename ColStride,
      typename RowOffset = Int<0>,
      typename ColOffset = Int<0>>
  METAL_FUNC static constexpr void store(
      const thread FragmentVector<T>& source,
      DestinationPointerType destination,
      RowStride row_stride,
      ColStride col_stride,
      RowOffset row_offset = {},
      ColOffset col_offset = {}
  ) {
    using U = PointerElementType<DestinationPointerType>;

    const short2 lane_coord = get_lane_coordinate();
    destination += lane_coord.y * row_stride + lane_coord.x * col_stride;

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENT_ROWS; i++) {
      const auto row = row_offset + i * ELEMENT_ROWS_JUMP;
      const auto col = col_offset;

      if constexpr (metal::is_same_v<ColStride, Int<1>>) {
        METAL_PRAGMA_UNROLL
        for (ushort j = 0; j < ELEMENT_COLS; j++) {
          destination[row * row_stride + col + j] =
              static_cast<U>(source[i * ELEMENT_COLS + j]);
        }
      } else {
        METAL_PRAGMA_UNROLL
        for (ushort j = 0; j < ELEMENT_COLS; j++) {
          destination[row * row_stride + (col + j) * col_stride] =
              static_cast<U>(source[i * ELEMENT_COLS + j]);
        }
      }
    }
  }

  template <
      typename T,
      typename DestinationPointerType,
      typename RowStride,
      typename ColStride,
      typename RowLimit,
      typename RowOffset = Int<0>,
      typename ColOffset = Int<0>>
  METAL_FUNC static constexpr void store_rows(
      const thread FragmentVector<T>& source,
      DestinationPointerType destination,
      RowStride row_stride,
      ColStride col_stride,
      RowLimit row_limit,
      RowOffset row_offset = {},
      ColOffset col_offset = {}
  ) {
    using U = PointerElementType<DestinationPointerType>;

    const short2 lane_coord = get_lane_coordinate();
    destination += lane_coord.y * row_stride + lane_coord.x * col_stride;
    auto local_row_limit = row_limit - lane_coord.y;

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENT_ROWS; i++) {
      const auto row = row_offset + i * ELEMENT_ROWS_JUMP;
      const auto col = col_offset;

      if (row < local_row_limit) {
        if constexpr (metal::is_same_v<ColStride, Int<1>>) {
          METAL_PRAGMA_UNROLL
          for (ushort j = 0; j < ELEMENT_COLS; j++) {
            destination[row * row_stride + col + j] =
                static_cast<U>(source[i * ELEMENT_COLS + j]);
          }
        } else {
          METAL_PRAGMA_UNROLL
          for (ushort j = 0; j < ELEMENT_COLS; j++) {
            destination[row * row_stride + (col + j) * col_stride] =
                static_cast<U>(source[i * ELEMENT_COLS + j]);
          }
        }
      }
    }
  }

  template <
      typename T,
      typename DestinationPointerType,
      typename RowStride,
      typename ColStride,
      typename RowLimit,
      typename ColLimit,
      typename RowOffset = Int<0>,
      typename ColOffset = Int<0>>
  METAL_FUNC static constexpr void store_safe(
      const thread FragmentVector<T>& source,
      DestinationPointerType destination,
      RowStride row_stride,
      ColStride col_stride,
      RowLimit row_limit,
      ColLimit col_limit,
      RowOffset row_offset = {},
      ColOffset col_offset = {}
  ) {
    using U = PointerElementType<DestinationPointerType>;

    const short2 lane_coord = get_lane_coordinate();
    destination += lane_coord.y * row_stride + lane_coord.x * col_stride;
    auto local_row_limit = row_limit - lane_coord.y;
    auto local_col_limit = col_limit - lane_coord.x;

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENT_ROWS; i++) {
      const auto row = row_offset + i * ELEMENT_ROWS_JUMP;
      const auto col = col_offset;

      METAL_PRAGMA_UNROLL
      for (ushort j = 0; j < ELEMENT_COLS; j++) {
        if (row < local_row_limit && (col + j) < local_col_limit) {
          destination[row * row_stride + (col + j) * col_stride] =
              static_cast<U>(source[i * ELEMENT_COLS + j]);
        }
      }
    }
  }

  template <
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
  METAL_FUNC static constexpr void mma(
      thread FragmentVector<CType>& output_col_0,
      thread FragmentVector<CType>& output_col_1,
      const thread FragmentVector<AType>& left,
      metal::bool_constant<transpose_a>,
      const thread FragmentVector<BType>& right_col_0,
      const thread FragmentVector<BType>& right_col_1,
      metal::bool_constant<transpose_b>
  ) {
    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        FRAGMENT_ROWS,
        2 * FRAGMENT_COLS,
        FRAGMENT_COLS,
        transpose_a,
        transpose_b,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
    );

    mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

    auto cooperative_left =
        matmul_op
            .template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto cooperative_right =
        matmul_op
            .template get_right_input_cooperative_tensor<AType, BType, CType>();
    auto cooperative_output =
        matmul_op.template get_destination_cooperative_tensor<
            decltype(cooperative_left),
            decltype(cooperative_right),
            CType>();

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENTS_PER_FRAG; i++) {
      cooperative_left[i] = left[i];
    }

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENTS_PER_FRAG; i++) {
      cooperative_right[i] = right_col_0[i];
      cooperative_right[ELEMENTS_PER_FRAG + i] = right_col_1[i];
    }

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENTS_PER_FRAG; i++) {
      cooperative_output[i] = output_col_0[i];
      cooperative_output[ELEMENTS_PER_FRAG + i] = output_col_1[i];
    }

    matmul_op.run(cooperative_left, cooperative_right, cooperative_output);

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENTS_PER_FRAG; i++) {
      output_col_0[i] = cooperative_output[i];
      output_col_1[i] = cooperative_output[ELEMENTS_PER_FRAG + i];
    }
  }

  template <
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
  METAL_FUNC static constexpr void mma(
      thread FragmentVector<CType>& output_row_0,
      thread FragmentVector<CType>& output_row_1,
      const thread FragmentVector<AType>& left_row_0,
      const thread FragmentVector<AType>& left_row_1,
      metal::bool_constant<transpose_a>,
      const thread FragmentVector<BType>& right,
      metal::bool_constant<transpose_b>
  ) {
    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        FRAGMENT_ROWS,
        2 * FRAGMENT_COLS,
        FRAGMENT_COLS,
        transpose_a,
        transpose_b,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
    );

    mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul_op;

    auto cooperative_left =
        matmul_op
            .template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto cooperative_right =
        matmul_op
            .template get_right_input_cooperative_tensor<AType, BType, CType>();
    auto cooperative_output =
        matmul_op.template get_destination_cooperative_tensor<
            decltype(cooperative_left),
            decltype(cooperative_right),
            CType>();

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENTS_PER_FRAG; i++) {
      cooperative_left[i] = left_row_0[i];
      cooperative_left[ELEMENTS_PER_FRAG + i] = left_row_1[i];
    }

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENTS_PER_FRAG; i++) {
      cooperative_right[i] = right[i];
    }

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENTS_PER_FRAG; i++) {
      cooperative_output[i] = output_row_0[i];
      cooperative_output[ELEMENTS_PER_FRAG + i] = output_row_1[i];
    }

    matmul_op.run(cooperative_left, cooperative_right, cooperative_output);

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < ELEMENTS_PER_FRAG; i++) {
      output_row_0[i] = cooperative_output[i];
      output_row_1[i] = cooperative_output[ELEMENTS_PER_FRAG + i];
    }
  }
};

} // namespace matmul
} // namespace uzu
