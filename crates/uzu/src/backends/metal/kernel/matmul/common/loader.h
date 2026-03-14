#pragma once

#include "defines.h"

using namespace metal;

namespace uzu {
namespace matmul {

///////////////////////////////////////////////////////////////////////////////
// Block Loader - loads tiles from device memory to threadgroup memory
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    short BLOCK_ROWS,
    short BLOCK_COLS,
    short destination_leading_dimension,
    short reduction_dim,
    short threadgroup_size,
    short alignment = 1,
    short num_reads = (BLOCK_COLS * BLOCK_ROWS) / (threadgroup_size),
    short THREAD_COLS = BLOCK_COLS / num_reads,
    short THREAD_ROWS = threadgroup_size / THREAD_COLS
>
struct BlockLoader {
  MTL_CONST short NUM_ROWS = (BLOCK_ROWS + THREAD_ROWS - 1) / THREAD_ROWS;
  MTL_CONST short VECTOR_SIZE = num_reads;

  // Leading dimension for source
  const int source_leading_dimension;
  const int tile_stride;

  // Thread location indices
  const short thread_idx;
  const short block_row;
  const short block_col;

  // threadgroup and device memory
  threadgroup T* destination;
  const device T* source;

  struct alignas(alignment * sizeof(T)) ReadVector {
    uint8_t v[sizeof(T) * VECTOR_SIZE];
  };

  /* Constructor */
  METAL_FUNC BlockLoader(
      const device T* source_,
      const int source_leading_dimension_,
      threadgroup T* destination_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]
  )
      : source_leading_dimension(source_leading_dimension_),
        tile_stride(
            reduction_dim ? BLOCK_COLS : BLOCK_ROWS * source_leading_dimension
        ),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        block_row(thread_idx / THREAD_COLS),
        block_col(VECTOR_SIZE * (thread_idx % THREAD_COLS)),
        destination(
            destination_ + block_row * destination_leading_dimension + block_col
        ),
        source(source_ + block_row * source_leading_dimension + block_col) {}

  /* Apply operation to threadgroup without bound checking */
  template <typename UnaryOp>
  METAL_FUNC void apply_inplace_op(thread const UnaryOp& op) const {
    PRAGMA_UNROLL
    for (short i = 0; i < BLOCK_ROWS; i += THREAD_ROWS) {
      PRAGMA_UNROLL
      for (short j = 0; j < VECTOR_SIZE; j++) {
        destination[i * destination_leading_dimension + j] =
            op.apply(destination[i * destination_leading_dimension + j]);
      }
    }
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unchecked() const {
    PRAGMA_UNROLL
    for (short i = 0; i < BLOCK_ROWS; i += THREAD_ROWS) {
      *((threadgroup ReadVector*)(&destination
                                      [i * destination_leading_dimension])) =
          *((const device ReadVector*)(&source[i * source_leading_dimension]));
    }
  }

  /* Load from device memory into threadgroup memory - with bound checking */
  METAL_FUNC void load_checked(short2 source_tile_dimensions) const {
    source_tile_dimensions =
        source_tile_dimensions - short2(block_col, block_row);

    // Skip loading if thread has no valid reads
    if (source_tile_dimensions.x <= 0 || source_tile_dimensions.y <= 0) {
      PRAGMA_UNROLL
      for (short i = 0; i < BLOCK_ROWS; i += THREAD_ROWS) {
        PRAGMA_UNROLL
        for (short j = 0; j < VECTOR_SIZE; j++) {
          destination[i * destination_leading_dimension + j] = T(0);
        }
      }
      return;
    }

    // Use fast thread memory for bound checks
    bool valid_indices[VECTOR_SIZE];
    T temp_values[VECTOR_SIZE];

    PRAGMA_UNROLL
    for (short i = 0; i < BLOCK_ROWS; i += THREAD_ROWS) {
      // Make sure valid_indices only contains valid indices
      PRAGMA_UNROLL
      for (short j = 0; j < VECTOR_SIZE; j++) {
        valid_indices[j] =
            (i < source_tile_dimensions.y) && (j < source_tile_dimensions.x);
      }

      // Read valid indices into temp_values
      PRAGMA_UNROLL
      for (short j = 0; j < VECTOR_SIZE; j++) {
        temp_values[j] =
            source[(valid_indices[j] ? i * source_leading_dimension + j : 0)];
      }

      // Zero out unneeded values
      PRAGMA_UNROLL
      for (short j = 0; j < VECTOR_SIZE; j++) {
        temp_values[j] = valid_indices[j] ? temp_values[j] : T(0);
      }

      // Copy values to threadgroup memory
      PRAGMA_UNROLL
      for (short j = 0; j < VECTOR_SIZE; j++) {
        destination[i * destination_leading_dimension + j] = temp_values[j];
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() { source += tile_stride; }
};

} // namespace matmul
} // namespace uzu
