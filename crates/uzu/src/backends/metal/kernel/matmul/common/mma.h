#pragma once

#include "defines.h"

using namespace metal;

namespace uzu {
namespace matmul {

///////////////////////////////////////////////////////////////////////////////
// MMA Fragment - 8x8 simdgroup matrix fragment
///////////////////////////////////////////////////////////////////////////////

template <typename T, int FRAG_ROWS_, int FRAG_COLS_>
struct BaseMMAFrag {
  static_assert(
      FRAG_ROWS_ == 8,
      "Only 8 x 8 fragment matrices are currently supported"
  );
  static_assert(
      FRAG_COLS_ == 8,
      "Only 8 x 8 fragment matrices are currently supported"
  );
};

template <typename T>
struct BaseMMAFrag<T, 8, 8> {
  MTL_CONST int FRAG_ROWS = 8;
  MTL_CONST int FRAG_COLS = 8;
  MTL_CONST int ELEMENTS_PER_FRAG = (FRAG_ROWS * FRAG_COLS) / 32;
  MTL_CONST int ELEMENT_ROWS = 1;
  MTL_CONST int ELEMENT_COLS = 2;

  static_assert(
      ELEMENT_ROWS * ELEMENT_COLS == ELEMENTS_PER_FRAG,
      "MMAFrag shape is not consistent with MMAFrag size"
  );

  typedef metal::simdgroup_matrix<T, FRAG_ROWS, FRAG_COLS> MatrixType;
  typedef metal::vec<T, ELEMENTS_PER_FRAG> FragmentType;

  METAL_FUNC static constexpr short2 get_coordinate(
      ushort simd_lane_id [[thread_index_in_simdgroup]]
  ) {
    const short quad_id = simd_lane_id / 4;
    const short fragment_row = (quad_id / 4) * 4 + ((simd_lane_id / 2) % 4);
    const short fragment_col = (quad_id % 4 / 2) * 4 + (simd_lane_id % 2) * 2;
    return short2{fragment_col, fragment_row};
  }

  template <typename SourcePointerType, typename StrideXType, typename StrideYType>
  METAL_FUNC static constexpr void load(
      thread FragmentType& destination,
      SourcePointerType source,
      StrideXType stride_x,
      StrideYType stride_y
  ) {
    PRAGMA_UNROLL
    for (short i = 0; i < ELEMENT_ROWS; i++) {
      PRAGMA_UNROLL
      for (short j = 0; j < ELEMENT_COLS; j++) {
        destination[i * ELEMENT_COLS + j] = static_cast<T>(source[i * stride_x + j * stride_y]);
      }
    }
  }

  template <
      typename SourcePointerType,
      typename StrideXType,
      typename StrideYType,
      typename LimitXType,
      typename LimitYType,
      typename OffsetXType,
      typename OffsetYType>
  METAL_FUNC static constexpr void load_checked(
      thread FragmentType& destination,
      SourcePointerType source,
      StrideXType stride_x,
      StrideYType stride_y,
      LimitXType limit_x,
      LimitYType limit_y,
      OffsetXType offset_x = 0,
      OffsetYType offset_y = 0
  ) {
    PRAGMA_UNROLL
    for (short i = 0; i < ELEMENT_ROWS; i++) {
      PRAGMA_UNROLL
      for (short j = 0; j < ELEMENT_COLS; j++) {
        if ((offset_x + i) < limit_x && (offset_y + j) < limit_y) {
          destination[i * ELEMENT_COLS + j] =
              static_cast<T>(source[(offset_x + i) * stride_x + (offset_y + j) * stride_y]);
        } else {
          destination[i * ELEMENT_COLS + j] = T(0);
        }
      }
    }
  }

  template <typename DestinationPointerType, typename StrideXType, typename StrideYType>
  METAL_FUNC static constexpr void store(
      const thread FragmentType& source,
      DestinationPointerType destination,
      StrideXType stride_x,
      StrideYType stride_y
  ) {
    using U = pointer_element_t<DestinationPointerType>;

    PRAGMA_UNROLL
    for (short i = 0; i < ELEMENT_ROWS; i++) {
      PRAGMA_UNROLL
      for (short j = 0; j < ELEMENT_COLS; j++) {
        destination[i * stride_x + j * stride_y] = static_cast<U>(source[i * ELEMENT_COLS + j]);
      }
    }
  }

  template <
      typename DestinationPointerType,
      typename StrideXType,
      typename StrideYType,
      typename LimitXType,
      typename LimitYType,
      typename OffsetXType,
      typename OffsetYType>
  METAL_FUNC static constexpr void store_checked(
      const thread FragmentType& source,
      DestinationPointerType destination,
      StrideXType stride_x,
      StrideYType stride_y,
      LimitXType limit_x,
      LimitYType limit_y,
      OffsetXType offset_x = 0,
      OffsetYType offset_y = 0
  ) {
    using U = pointer_element_t<DestinationPointerType>;

    PRAGMA_UNROLL
    for (short i = 0; i < ELEMENT_ROWS; i++) {
      PRAGMA_UNROLL
      for (short j = 0; j < ELEMENT_COLS; j++) {
        if ((offset_x + i) < limit_x && (offset_y + j) < limit_y) {
          destination[(offset_x + i) * stride_x + (offset_y + j) * stride_y] =
              static_cast<U>(source[i * ELEMENT_COLS + j]);
        }
      }
    }
  }

  METAL_FUNC static constexpr void mma(
      thread FragmentType& D,
      thread FragmentType& A,
      thread FragmentType& B,
      thread FragmentType& C
  ) {
    MatrixType D_mat;
    MatrixType A_mat;
    MatrixType B_mat;
    MatrixType C_mat;

    reinterpret_cast<thread FragmentType&>(A_mat.thread_elements()) = A;
    reinterpret_cast<thread FragmentType&>(B_mat.thread_elements()) = B;
    reinterpret_cast<thread FragmentType&>(C_mat.thread_elements()) = C;

    mma(D_mat, A_mat, B_mat, C_mat);

    D = reinterpret_cast<thread FragmentType&>(D_mat.thread_elements());
  }

  METAL_FUNC static constexpr void mma(
      thread MatrixType& D,
      thread MatrixType& A,
      thread MatrixType& B,
      thread MatrixType& C
  ) {
    simdgroup_multiply_accumulate(D, A, B, C);
  }
};

///////////////////////////////////////////////////////////////////////////////
// MMA Tile - collection of MMA fragments
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    int TILE_ROWS_,
    int TILE_COLS_,
    class MMAFrag_ = BaseMMAFrag<T, 8, 8>>
struct MMATile {
  using MMAFragType = MMAFrag_;
  using ElementType = T;
  MTL_CONST int FRAG_ROWS = MMAFragType::FRAG_ROWS;
  MTL_CONST int FRAG_COLS = MMAFragType::FRAG_COLS;
  MTL_CONST int ELEMENTS_PER_FRAG = MMAFragType::ELEMENTS_PER_FRAG;

  MTL_CONST int TILE_ROWS = TILE_ROWS_;
  MTL_CONST int TILE_COLS = TILE_COLS_;

  MTL_CONST int ROWS = TILE_ROWS * FRAG_ROWS;
  MTL_CONST int COLS = TILE_COLS * FRAG_COLS;

  MTL_CONST int NUM_FRAGS = TILE_ROWS * TILE_COLS;
  MTL_CONST int ELEMENTS_PER_TILE = NUM_FRAGS * ELEMENTS_PER_FRAG;

  typedef typename MMAFragType::MatrixType MatrixType;
  typedef typename MMAFragType::FragmentType FragmentType;

  FragmentType value_fragments[NUM_FRAGS] = {FragmentType(0)};

  METAL_FUNC MMATile() thread {}

  METAL_FUNC constexpr void clear() {
    PRAGMA_UNROLL
    for (short i = 0; i < NUM_FRAGS; ++i) {
      value_fragments[i] = FragmentType(0);
    }
  }

  METAL_FUNC constexpr thread FragmentType& fragment_at(const short i, const short j) {
    return value_fragments[i * TILE_COLS + j];
  }

  METAL_FUNC constexpr const thread FragmentType& fragment_at(
      const short i,
      const short j
  ) const {
    return value_fragments[i * TILE_COLS + j];
  }

  METAL_FUNC thread ElementType* elements() {
    return reinterpret_cast<thread ElementType*>(value_fragments);
  }

  METAL_FUNC const thread ElementType* elements() const {
    return reinterpret_cast<const thread ElementType*>(value_fragments);
  }

  template <typename U, int warp_stride_x, int warp_stride_y, int stride_x, int stride_y>
  METAL_FUNC void load(const threadgroup U* source) {
    PRAGMA_UNROLL
    for (short i = 0; i < TILE_ROWS; ++i) {
      PRAGMA_UNROLL
      for (short j = 0; j < TILE_COLS; ++j) {
        MMAFragType::load(
            fragment_at(i, j),
            &(
                source[(i * FRAG_ROWS) * warp_stride_x * stride_x +
                    (j * FRAG_COLS) * warp_stride_y * stride_y]
            ),
            stride_x,
            stride_y
        );
      }
    }
  }

  template <typename U, int warp_stride_x, int warp_stride_y>
  METAL_FUNC void store(device U* destination, const int leading_dimension) const {
    PRAGMA_UNROLL
    for (short i = 0; i < TILE_ROWS; ++i) {
      PRAGMA_UNROLL
      for (short j = 0; j < TILE_COLS; ++j) {
        MMAFragType::store(
            fragment_at(i, j),
            &(destination[(i * FRAG_ROWS) * warp_stride_x * leading_dimension + (j * FRAG_COLS) * warp_stride_y]),
            leading_dimension,
            1
        );
      }
    }
  }

  template <typename U, int warp_stride_x, int warp_stride_y>
  METAL_FUNC void store_checked(
      device U* destination,
      const int leading_dimension,
      const short2 destination_tile_dimensions
  ) const {
    PRAGMA_UNROLL
    for (int i = 0; i < TILE_ROWS; ++i) {
      PRAGMA_UNROLL
      for (int j = 0; j < TILE_COLS; ++j) {
        MMAFragType::store_checked(
            fragment_at(i, j),
            destination,
            leading_dimension,
            1,
            destination_tile_dimensions.y,
            destination_tile_dimensions.x,
            (i * FRAG_ROWS) * warp_stride_x,
            (j * FRAG_COLS) * warp_stride_y
        );
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
// Tile matrix multiply-accumulate
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U, int M, int N, int K>
METAL_FUNC void tile_multiply_add(
    thread MMATile<T, M, N>& D,
    thread MMATile<U, M, K>& A,
    thread MMATile<U, K, N>& B,
    thread MMATile<T, M, N>& C
) {
  PRAGMA_UNROLL
  for (short m = 0; m < M; ++m) {
    PRAGMA_UNROLL
    for (short n = 0; n < N; ++n) {
      short serpentine_n = (m % 2) ? (N - 1 - n) : n;
      PRAGMA_UNROLL
      for (short k = 0; k < K; ++k) {
        MMATile<T, M, N>::MMAFragType::mma(
            D.fragment_at(m, serpentine_n),
            A.fragment_at(m, k),
            B.fragment_at(k, serpentine_n),
            C.fragment_at(m, serpentine_n)
        );
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Block MMA - manages the GEMM computation for a threadgroup
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    typename U,
    int BLOCK_M,
    int BLOCK_N,
    int BLOCK_K,
    int WARPS_M,
    int WARPS_N,
    bool transpose_a,
    bool transpose_b,
    short threadgroup_leading_dim_a,
    short threadgroup_leading_dim_b,
    typename AccumulatorType = float,
    typename Epilogue = TransformNone<U, AccumulatorType>>
struct BlockMMA {
  // MMAFrag size
  MTL_CONST short FRAG_SIZE = 8;
  using MMAFragAccumulatorType = BaseMMAFrag<AccumulatorType, FRAG_SIZE, FRAG_SIZE>;

  // Warp tile simdgroup matrix strides along M
  MTL_CONST short TILES_M_STRIDE = FRAG_SIZE * WARPS_M;
  // Warp tile simdgroup matrix strides along N
  MTL_CONST short TILES_N_STRIDE = FRAG_SIZE * WARPS_N;

  // Warp tile size along M
  MTL_CONST short TILES_M = BLOCK_M / (FRAG_SIZE * WARPS_M);
  // Warp tile size along N
  MTL_CONST short TILES_N = BLOCK_N / (FRAG_SIZE * WARPS_N);

  // Threadgroup A strides
  MTL_CONST short left_stride_m = transpose_a ? 1 : threadgroup_leading_dim_a; // M
  MTL_CONST short left_stride_k = transpose_a ? threadgroup_leading_dim_a : 1; // K

  // Threadgroup B strides
  MTL_CONST short right_stride_k = transpose_b ? 1 : threadgroup_leading_dim_b; // K
  MTL_CONST short right_stride_n = transpose_b ? threadgroup_leading_dim_b : 1; // N

  // Threadgroup strides along K
  MTL_CONST short tile_stride_a = FRAG_SIZE * left_stride_k;
  MTL_CONST short tile_stride_b = FRAG_SIZE * right_stride_k;

  // Simdgroup matrices
  MMATile<AccumulatorType, TILES_M, 1, MMAFragAccumulatorType> left_tile;
  MMATile<AccumulatorType, 1, TILES_N, MMAFragAccumulatorType> right_tile;
  MMATile<AccumulatorType, TILES_M, TILES_N, MMAFragAccumulatorType> accumulator_tile;

  // Offsets within threadgroup
  short simdgroup_row;
  short simdgroup_col;

  short left_shared_offset;
  short right_shared_offset;

  /* Constructor */
  METAL_FUNC BlockMMA(
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]
  ) {
    // Determine thread position in simdgroup matrix
    short tile_row_offset = FRAG_SIZE * (simd_group_id / WARPS_N);
    short tile_col_offset = FRAG_SIZE * (simd_group_id % WARPS_N);

    short2 simd_coordinate = MMAFragAccumulatorType::get_coordinate(simd_lane_id);
    simdgroup_row = simd_coordinate.y;
    simdgroup_col = simd_coordinate.x;

    // Determine thread and simdgroup offset
    left_shared_offset = (tile_row_offset + simdgroup_row) * left_stride_m + (simdgroup_col) * left_stride_k; // M, K
    right_shared_offset = (simdgroup_row) * right_stride_k + (tile_col_offset + simdgroup_col) * right_stride_n; // K, N

    simdgroup_row += tile_row_offset;
    simdgroup_col += tile_col_offset;
  }

  /* (BLOCK_M, BLOCK_K) X (BLOCK_K, BLOCK_N) multiply accumulate function */
  METAL_FUNC void mma(const threadgroup T* left_shared, const threadgroup T* right_shared) {
    // Adjust for simdgroup and thread location
    left_shared += left_shared_offset;
    right_shared += right_shared_offset;

    // Iterate over BLOCK_K in blocks of FRAG_SIZE
    PRAGMA_UNROLL
    for (short k_inner = 0; k_inner < BLOCK_K; k_inner += FRAG_SIZE) {
      simdgroup_barrier(mem_flags::mem_none);

      left_tile.template load<T, WARPS_M, 1, left_stride_m, left_stride_k>(left_shared);

      simdgroup_barrier(mem_flags::mem_none);

      right_tile.template load<T, 1, WARPS_N, right_stride_k, right_stride_n>(right_shared);

      simdgroup_barrier(mem_flags::mem_none);

      tile_multiply_add(accumulator_tile, left_tile, right_tile, accumulator_tile);

      // Progress to next simdgroup tile
      left_shared += tile_stride_a;
      right_shared += tile_stride_b;
    }
  }

  /* Store results from simdgroup_matrix results into device memory */
  METAL_FUNC void store_result(device U* output_matrix, const int leading_dim_d) {
    // Apply epilogue
    PRAGMA_UNROLL
    for (short i = 0; i < decltype(accumulator_tile)::ELEMENTS_PER_TILE; i++) {
      accumulator_tile.elements()[i] = Epilogue::apply(accumulator_tile.elements()[i]);
    }

    // Adjust for simdgroup and thread location
    output_matrix += simdgroup_row * leading_dim_d + simdgroup_col;

    accumulator_tile.template store<U, WARPS_M, WARPS_N>(output_matrix, leading_dim_d);
  }

  METAL_FUNC void store_result_checked(
      device U* output_matrix,
      const int leading_dim_d,
      short2 destination_tile_dimensions
  ) {
    // Apply epilogue
    PRAGMA_UNROLL
    for (short i = 0; i < decltype(accumulator_tile)::ELEMENTS_PER_TILE; i++) {
      accumulator_tile.elements()[i] = Epilogue::apply(accumulator_tile.elements()[i]);
    }

    // Adjust for simdgroup and thread location
    output_matrix += simdgroup_row * leading_dim_d + simdgroup_col;
    destination_tile_dimensions -= short2(simdgroup_col, simdgroup_row);

    if (destination_tile_dimensions.x <= 0 || destination_tile_dimensions.y <= 0)
      return;

    accumulator_tile.template store_checked<U, WARPS_M, WARPS_N>(output_matrix, leading_dim_d, destination_tile_dimensions);
  }

  /* Apply epilogue with accumulate_source matrix */
  template <typename EpilogueOpType>
  METAL_FUNC void apply_epilogue(
      const device U* accumulate_source,
      const int leading_dim_c,
      const int col_stride_c,
      thread const EpilogueOpType& epilogue_op
  ) {
    const device U* accumulate_source_ptr = accumulate_source + simdgroup_row * leading_dim_c + simdgroup_col * col_stride_c;

    PRAGMA_UNROLL
    for (short i = 0; i < TILES_M; i++) {
      PRAGMA_UNROLL
      for (short j = 0; j < TILES_N; j++) {
        thread auto& frag = accumulator_tile.fragment_at(i, j);
        PRAGMA_UNROLL
        for (short k = 0; k < decltype(accumulator_tile)::ELEMENTS_PER_FRAG; k++) {
          short row_offset = (i * FRAG_SIZE * WARPS_M);
          short col_offset = (j * FRAG_SIZE * WARPS_N);
          // Get the accumulate_source element (accounting for fragment layout)
          U c_val = accumulate_source_ptr[row_offset * leading_dim_c + col_offset * col_stride_c + k];
          frag[k] = epilogue_op.apply(frag[k], static_cast<AccumulatorType>(c_val));
        }
      }
    }
  }

  template <typename EpilogueOpType>
  METAL_FUNC void apply_epilogue_checked(
      const device U* accumulate_source,
      const int leading_dim_c,
      const int col_stride_c,
      short2 tile_dimensions,
      thread const EpilogueOpType& epilogue_op
  ) {
    const device U* accumulate_source_ptr = accumulate_source + simdgroup_row * leading_dim_c + simdgroup_col * col_stride_c;
    tile_dimensions -= short2(simdgroup_col, simdgroup_row);

    PRAGMA_UNROLL
    for (short i = 0; i < TILES_M; i++) {
      PRAGMA_UNROLL
      for (short j = 0; j < TILES_N; j++) {
        thread auto& frag = accumulator_tile.fragment_at(i, j);
        short row_offset = (i * FRAG_SIZE * WARPS_M);
        short col_offset = (j * FRAG_SIZE * WARPS_N);
        PRAGMA_UNROLL
        for (short k = 0; k < decltype(accumulator_tile)::ELEMENTS_PER_FRAG; k++) {
          if (row_offset < tile_dimensions.y && col_offset + k < tile_dimensions.x) {
            U c_val = accumulate_source_ptr[row_offset * leading_dim_c + col_offset * col_stride_c + k];
            frag[k] = epilogue_op.apply(frag[k], static_cast<AccumulatorType>(c_val));
          }
        }
      }
    }
  }
};

} // namespace matmul
} // namespace uzu
