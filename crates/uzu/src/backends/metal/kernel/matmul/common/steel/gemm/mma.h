

#pragma once

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

#include "../../defines.h"
#include "../utils/type_traits.h"
#include "transforms.h"

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// MMA helper
///////////////////////////////////////////////////////////////////////////////

namespace steel {

// Max tile dimensions across all configurations.
// GEMM: TILES_M up to 8 (64/(8*1)), TILES_N up to 4 (64/(8*2))
// Split-K: TILES_M up to 1, TILES_N up to 2
// tile_multiply_add K dim is always 1
#define UZU_MAX_TM 8
#define UZU_MAX_TN 4
#define UZU_MAX_CTILE_FRAGS (UZU_MAX_TM * UZU_MAX_TN) // 32

template <typename T>
struct BaseMMAFrag {
  MTL_CONST int FRAG_ROWS = 8;
  MTL_CONST int FRAG_COLS = 8;

  MTL_CONST int ELEMENTS_PER_FRAG = (FRAG_ROWS * FRAG_COLS) / 32;

  MTL_CONST int ELEMENT_ROWS = 1;
  MTL_CONST int ELEMENT_COLS = 2;

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

  template <typename SourcePointerType>
  METAL_FUNC static constexpr void load(
      thread FragmentType& destination,
      SourcePointerType source,
      int stride_x,
      int stride_y
  ) {
    PRAGMA_UNROLL
    for (short i = 0; i < ELEMENT_ROWS; i++) {
      PRAGMA_UNROLL
      for (short j = 0; j < ELEMENT_COLS; j++) {
        destination[i * ELEMENT_COLS + j] = static_cast<T>(source[i * stride_x + j * stride_y]);
      }
    }
  }

  template <typename SourcePointerType>
  METAL_FUNC static constexpr void load_checked(
      thread FragmentType& destination,
      SourcePointerType source,
      int stride_x,
      int stride_y,
      int limit_x,
      int limit_y,
      int offset_x = 0,
      int offset_y = 0
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

  template <typename DestinationPointerType>
  METAL_FUNC static constexpr void store(
      const thread FragmentType& source,
      DestinationPointerType destination,
      int stride_x,
      int stride_y
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

  template <typename DestinationPointerType>
  METAL_FUNC static constexpr void store_checked(
      const thread FragmentType& source,
      DestinationPointerType destination,
      int stride_x,
      int stride_y,
      int limit_x,
      int limit_y,
      int offset_x = 0,
      int offset_y = 0
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

  template <typename DestinationPointerType>
  METAL_FUNC static constexpr void store_slice(
      const thread FragmentType& source,
      DestinationPointerType destination,
      int stride_x,
      int stride_y,
      int start_x,
      int stop_x,
      int start_y,
      int stop_y,
      int offset_x = 0,
      int offset_y = 0
  ) {
    using U = pointer_element_t<DestinationPointerType>;

    PRAGMA_UNROLL
    for (short i = 0; i < ELEMENT_ROWS; i++) {
      PRAGMA_UNROLL
      for (short j = 0; j < ELEMENT_COLS; j++) {
        if ((offset_x + i) < stop_x && (offset_x + i) >= start_x &&
            (offset_y + j) < stop_y && (offset_y + j) >= start_y) {
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

template <typename T>
struct MMATile {
  using MMAFragType = BaseMMAFrag<T>;
  using ElementType = T;
  MTL_CONST int FRAG_ROWS = MMAFragType::FRAG_ROWS;
  MTL_CONST int FRAG_COLS = MMAFragType::FRAG_COLS;
  MTL_CONST int ELEMENTS_PER_FRAG = MMAFragType::ELEMENTS_PER_FRAG;

  typedef typename MMAFragType::MatrixType MatrixType;
  typedef typename MMAFragType::FragmentType FragmentType;

  // Max-sized array -- runtime TILE_ROWS/TILE_COLS control how much is used
  FragmentType value_fragments[UZU_MAX_CTILE_FRAGS] = {};

  short TILE_ROWS;
  short TILE_COLS;
  short NUM_FRAGS;

  METAL_FUNC MMATile() thread : TILE_ROWS(0), TILE_COLS(0), NUM_FRAGS(0) {}

  METAL_FUNC MMATile(short tile_rows, short tile_cols) thread
      : TILE_ROWS(tile_rows),
        TILE_COLS(tile_cols),
        NUM_FRAGS(tile_rows * tile_cols) {}

  METAL_FUNC constexpr void clear() {
    PRAGMA_UNROLL
    for (short i = 0; i < UZU_MAX_CTILE_FRAGS; ++i) {
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

  METAL_FUNC short elements_per_tile() const { return NUM_FRAGS * ELEMENTS_PER_FRAG; }

  template <typename U>
  METAL_FUNC void load(
      const threadgroup U* source,
      int warp_stride_x,
      int warp_stride_y,
      int stride_x,
      int stride_y
  ) {
    for (short i = 0; i < TILE_ROWS; ++i) {
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

  template <typename U>
  METAL_FUNC void store(
      threadgroup U* destination,
      int warp_stride_x,
      int warp_stride_y,
      int stride_x,
      int stride_y
  ) const {
    for (short i = 0; i < TILE_ROWS; ++i) {
      for (short j = 0; j < TILE_COLS; ++j) {
        MMAFragType::store(
            fragment_at(i, j),
            &(
                destination[(i * FRAG_ROWS) * warp_stride_x * stride_x +
                    (j * FRAG_COLS) * warp_stride_y * stride_y]
            ),
            stride_x,
            stride_y
        );
      }
    }
  }

  template <typename U>
  METAL_FUNC void load(const device U* source, const int leading_dimension, int warp_stride_x, int warp_stride_y) {
    for (short i = 0; i < TILE_ROWS; ++i) {
      for (short j = 0; j < TILE_COLS; ++j) {
        MMAFragType::load(
            fragment_at(i, j),
            &(source[(i * FRAG_ROWS) * warp_stride_x * leading_dimension + (j * FRAG_COLS) * warp_stride_y]),
            leading_dimension,
            1
        );
      }
    }
  }

  template <typename U>
  METAL_FUNC void store(device U* destination, const int leading_dimension, int warp_stride_x, int warp_stride_y) const {
    for (short i = 0; i < TILE_ROWS; ++i) {
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

  template <typename U>
  METAL_FUNC void load_checked(
      const device U* source,
      const int leading_dimension,
      const short2 source_tile_dimensions,
      int warp_stride_x,
      int warp_stride_y
  ) {
    for (int i = 0; i < TILE_ROWS; ++i) {
      for (int j = 0; j < TILE_COLS; ++j) {
        MMAFragType::load_checked(
            fragment_at(i, j),
            source,
            leading_dimension,
            1,
            source_tile_dimensions.y,
            source_tile_dimensions.x,
            (i * FRAG_ROWS) * warp_stride_x,
            (j * FRAG_COLS) * warp_stride_y
        );
      }
    }
  }

  template <typename U>
  METAL_FUNC void store_checked(
      device U* destination,
      const int leading_dimension,
      const short2 destination_tile_dimensions,
      int warp_stride_x,
      int warp_stride_y
  ) const {
    for (int i = 0; i < TILE_ROWS; ++i) {
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

  template <typename U>
  METAL_FUNC void store_slice(
      device U* destination,
      const int leading_dimension,
      const short2 start,
      const short2 stop,
      int warp_stride_x,
      int warp_stride_y
  ) const {
    for (int i = 0; i < TILE_ROWS; ++i) {
      for (int j = 0; j < TILE_COLS; ++j) {
        MMAFragType::store_slice(
            fragment_at(i, j),
            destination,
            leading_dimension,
            1,
            start.y,
            stop.y,
            start.x,
            stop.x,
            (i * FRAG_ROWS) * warp_stride_x,
            (j * FRAG_COLS) * warp_stride_y
        );
      }
    }
  }
};

template <typename T, typename U>
METAL_FUNC void tile_multiply_add(
    thread MMATile<T>& D,
    thread MMATile<U>& A,
    thread MMATile<U>& B,
    thread MMATile<T>& C
) {
  for (short m = 0; m < A.TILE_ROWS; ++m) {
    for (short n = 0; n < B.TILE_COLS; ++n) {
      short serpentine_n = (m % 2) ? (B.TILE_COLS - 1 - n) : n;
      for (short k = 0; k < A.TILE_COLS; ++k) {
        BaseMMAFrag<T>::mma(
            D.fragment_at(m, serpentine_n),
            A.fragment_at(m, k),
            B.fragment_at(k, serpentine_n),
            C.fragment_at(m, serpentine_n)
        );
      }
    }
  }
}

template <typename InputType>
struct TransformNone<complex64_t, InputType> {
  static METAL_FUNC complex64_t apply(complex64_t x) { return x; }
  static METAL_FUNC complex64_t apply(complex64_t x, complex64_t) { return x; }
};

template <
    typename T,
    typename U,
    typename AccumulatorType = float,
    typename Epilogue = TransformNone<U, AccumulatorType>>
struct BlockMMA {
  MTL_CONST short FRAG_SIZE = 8;
  using MMAFragAccumulatorType = BaseMMAFrag<AccumulatorType>;

  // Tile params (set at construction)
  short BLOCK_M, BLOCK_N, BLOCK_K, WARPS_M, WARPS_N;
  bool transpose_a, transpose_b;
  short threadgroup_leading_dim_a, threadgroup_leading_dim_b;

  // Derived values
  short TILES_M_STRIDE, TILES_N_STRIDE;
  short TILES_M, TILES_N;
  short left_stride_m, left_stride_k;
  short right_stride_k, right_stride_n;
  short tile_stride_a, tile_stride_b;

  // Simdgroup matrices
  MMATile<AccumulatorType> left_tile;
  MMATile<AccumulatorType> right_tile;
  MMATile<AccumulatorType> accumulator_tile;

  // Offsets within threadgroup
  short simdgroup_row;
  short simdgroup_col;
  short left_shared_offset;
  short right_shared_offset;

  /* Constructor */
  METAL_FUNC BlockMMA(
      ushort simd_group_id,
      ushort simd_lane_id,
      short BLOCK_M_,
      short BLOCK_N_,
      short BLOCK_K_,
      short WARPS_M_,
      short WARPS_N_,
      bool transpose_a_,
      bool transpose_b_,
      short threadgroup_leading_dim_a_,
      short threadgroup_leading_dim_b_
  )
      : BLOCK_M(BLOCK_M_), BLOCK_N(BLOCK_N_), BLOCK_K(BLOCK_K_),
        WARPS_M(WARPS_M_), WARPS_N(WARPS_N_),
        transpose_a(transpose_a_), transpose_b(transpose_b_),
        threadgroup_leading_dim_a(threadgroup_leading_dim_a_),
        threadgroup_leading_dim_b(threadgroup_leading_dim_b_),
        TILES_M_STRIDE(FRAG_SIZE * WARPS_M_), TILES_N_STRIDE(FRAG_SIZE * WARPS_N_),
        TILES_M(BLOCK_M_ / (FRAG_SIZE * WARPS_M_)), TILES_N(BLOCK_N_ / (FRAG_SIZE * WARPS_N_)),
        left_stride_m(transpose_a_ ? 1 : threadgroup_leading_dim_a_),
        left_stride_k(transpose_a_ ? threadgroup_leading_dim_a_ : 1),
        right_stride_k(transpose_b_ ? 1 : threadgroup_leading_dim_b_),
        right_stride_n(transpose_b_ ? threadgroup_leading_dim_b_ : 1),
        tile_stride_a(FRAG_SIZE * (transpose_a_ ? threadgroup_leading_dim_a_ : 1)),
        tile_stride_b(FRAG_SIZE * (transpose_b_ ? 1 : threadgroup_leading_dim_b_)),
        left_tile(TILES_M, 1),
        right_tile(1, TILES_N),
        accumulator_tile(TILES_M, TILES_N) {
    short tile_row_offset = FRAG_SIZE * (simd_group_id / WARPS_N);
    short tile_col_offset = FRAG_SIZE * (simd_group_id % WARPS_N);

    short2 simd_coordinate = MMAFragAccumulatorType::get_coordinate(simd_lane_id);
    simdgroup_row = simd_coordinate.y;
    simdgroup_col = simd_coordinate.x;

    left_shared_offset = (tile_row_offset + simdgroup_row) * left_stride_m + (simdgroup_col) * left_stride_k;
    right_shared_offset = (simdgroup_row) * right_stride_k + (tile_col_offset + simdgroup_col) * right_stride_n;

    simdgroup_row += tile_row_offset;
    simdgroup_col += tile_col_offset;
  }

  /* (BLOCK_M, BLOCK_K) X (BLOCK_K, BLOCK_N) multiply accumulate function */
  METAL_FUNC void mma(const threadgroup T* left_shared, const threadgroup T* right_shared) {
    left_shared += left_shared_offset;
    right_shared += right_shared_offset;

    for (short k_inner = 0; k_inner < BLOCK_K; k_inner += FRAG_SIZE) {
      simdgroup_barrier(mem_flags::mem_none);
      left_tile.load(left_shared, WARPS_M, 1, left_stride_m, left_stride_k);
      simdgroup_barrier(mem_flags::mem_none);
      right_tile.load(right_shared, 1, WARPS_N, right_stride_k, right_stride_n);
      simdgroup_barrier(mem_flags::mem_none);
      tile_multiply_add(accumulator_tile, left_tile, right_tile, accumulator_tile);
      left_shared += tile_stride_a;
      right_shared += tile_stride_b;
    }
  }

  /* Store results from simdgroup_matrix results into device memory */
  METAL_FUNC void store_result(device U* output_matrix, const int leading_dim_d) {
    for (short i = 0; i < accumulator_tile.elements_per_tile(); i++) {
      accumulator_tile.elements()[i] = Epilogue::apply(accumulator_tile.elements()[i]);
    }
    output_matrix += simdgroup_row * leading_dim_d + simdgroup_col;
    accumulator_tile.store(output_matrix, leading_dim_d, WARPS_M, WARPS_N);
  }

  METAL_FUNC void store_result_checked(
      device U* output_matrix,
      const int leading_dim_d,
      short2 destination_tile_dimensions
  ) {
    for (short i = 0; i < accumulator_tile.elements_per_tile(); i++) {
      accumulator_tile.elements()[i] = Epilogue::apply(accumulator_tile.elements()[i]);
    }
    output_matrix += simdgroup_row * leading_dim_d + simdgroup_col;
    destination_tile_dimensions -= short2(simdgroup_col, simdgroup_row);
    if (destination_tile_dimensions.x <= 0 || destination_tile_dimensions.y <= 0)
      return;
    accumulator_tile.store_checked(output_matrix, leading_dim_d, destination_tile_dimensions, WARPS_M, WARPS_N);
  }

  /* Apply epilogue */
  template <typename BinaryEpilogueType>
  METAL_FUNC void apply_epilogue(
      const device U* accumulate_source,
      const int leading_dim_c,
      const int col_stride_c,
      thread const BinaryEpilogueType& epilogue_op
  ) {
    accumulate_source += (simdgroup_row) * leading_dim_c + (simdgroup_col) * col_stride_c;

    for (short i = 0; i < TILES_M; i++) {
      for (short j = 0; j < TILES_N; j++) {
        thread auto& accumulator = accumulator_tile.fragment_at(i, j);
        int offset_c = (i * TILES_M_STRIDE) * leading_dim_c + (j * TILES_N_STRIDE) * col_stride_c;

        PRAGMA_UNROLL
        for (short k = 0; k < ELEMENTS_PER_FRAG; k++) {
          accumulator[k] = epilogue_op.apply(accumulator[k], accumulate_source[offset_c + k * col_stride_c]);
        }
      }
    }
  }

  MTL_CONST short ELEMENTS_PER_FRAG = MMAFragAccumulatorType::ELEMENTS_PER_FRAG;

  /* Apply epilogue checked */
  template <typename BinaryEpilogueType>
  METAL_FUNC void apply_epilogue_checked(
      const device U* accumulate_source,
      const int leading_dim_c,
      const int col_stride_c,
      short2 destination_tile_dimensions,
      thread const BinaryEpilogueType& epilogue_op
  ) {
    accumulate_source += (simdgroup_row) * leading_dim_c + (simdgroup_col) * col_stride_c;
    destination_tile_dimensions -= short2(simdgroup_col, simdgroup_row);

    if (destination_tile_dimensions.x <= 0 || destination_tile_dimensions.y <= 0)
      return;

    for (short i = 0; i < TILES_M; i++) {
      for (short j = 0; j < TILES_N; j++) {
        thread auto& accumulator = accumulator_tile.fragment_at(i, j);
        int offset_c = (i * TILES_M_STRIDE) * leading_dim_c + (j * TILES_N_STRIDE) * col_stride_c;

        U c_elems[2] = {0};

        PRAGMA_UNROLL
        for (short k = 0; k < ELEMENTS_PER_FRAG; k++) {
          if ((j * TILES_N_STRIDE + k) < destination_tile_dimensions.x) {
            c_elems[k] = accumulate_source[offset_c + k * col_stride_c];
          }
        }

        PRAGMA_UNROLL
        for (short k = 0; k < ELEMENTS_PER_FRAG; k++) {
          accumulator[k] = epilogue_op.apply(accumulator[k], c_elems[k]);
        }
      }
    }
  }

  /* Store results with epilogue from accumulate_source */
  METAL_FUNC void store_result(
      device U* output_matrix,
      const int leading_dim_d,
      const device U* accumulate_source,
      const int leading_dim_c,
      const int col_stride_c,
      thread const Epilogue& epilogue_op
  ) const {
    accumulate_source += (simdgroup_row) * leading_dim_c + (simdgroup_col) * col_stride_c;
    output_matrix += (simdgroup_row) * leading_dim_d + simdgroup_col;

    for (short i = 0; i < TILES_M; i++) {
      for (short j = 0; j < TILES_N; j++) {
        thread const auto& accumulator = accumulator_tile.fragment_at(i, j);
        int offset_c = (i * TILES_M_STRIDE) * leading_dim_c + (j * TILES_N_STRIDE) * col_stride_c;
        int offset_d = (i * TILES_M_STRIDE) * leading_dim_d + (j * TILES_N_STRIDE);

        PRAGMA_UNROLL
        for (short k = 0; k < ELEMENTS_PER_FRAG; k++) {
          output_matrix[offset_d + k] = epilogue_op.apply(accumulator[k], accumulate_source[offset_c + k * col_stride_c]);
        }
      }
    }
  }

  METAL_FUNC void store_result_checked(
      device U* output_matrix,
      const int leading_dim_d,
      const device U* accumulate_source,
      const int leading_dim_c,
      const int col_stride_c,
      short2 destination_tile_dimensions,
      thread const Epilogue& epilogue_op
  ) const {
    accumulate_source += (simdgroup_row) * leading_dim_c + (simdgroup_col) * col_stride_c;
    output_matrix += (simdgroup_row) * leading_dim_d + simdgroup_col;
    destination_tile_dimensions -= short2(simdgroup_col, simdgroup_row);

    if (destination_tile_dimensions.x <= 0 || destination_tile_dimensions.y <= 0)
      return;

    for (int i = 0; i < TILES_M; i++) {
      if (i * TILES_M_STRIDE < destination_tile_dimensions.y) {
        for (int j = 0; j < TILES_N; j++) {
          thread const auto& accumulator = accumulator_tile.fragment_at(i, j);
          int offset_c = (i * TILES_M_STRIDE) * leading_dim_c + (j * TILES_N_STRIDE) * col_stride_c;
          int offset_d = (i * TILES_M_STRIDE) * leading_dim_d + (j * TILES_N_STRIDE);

          PRAGMA_UNROLL
          for (short k = 0; k < ELEMENTS_PER_FRAG; k++) {
            if ((j * TILES_N_STRIDE + k) < destination_tile_dimensions.x) {
              output_matrix[offset_d + k] =
                  epilogue_op.apply(accumulator[k], accumulate_source[offset_c + k * col_stride_c]);
            }
          }
        }
      }
    }
  }
};

} // namespace steel
