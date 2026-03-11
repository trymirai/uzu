

#pragma once

#include "../../defines.h"

///////////////////////////////////////////////////////////////////////////////
// Loading helper
///////////////////////////////////////////////////////////////////////////////

namespace steel {

template <typename T>
struct BlockLoader {
  // Tile params
  const short BLOCK_ROWS;
  const short destination_leading_dimension;
  const short vector_size;
  const short THREAD_ROWS;

  // Leading dimension for source
  const int source_leading_dimension;
  const int tile_stride;

  // Thread location indices
  const short block_row;
  const short block_col;

  // threadgroup and device memory
  threadgroup T* destination;
  const device T* source;

  /* Constructor */
  METAL_FUNC BlockLoader(
      const device T* source_,
      const int source_leading_dimension_,
      threadgroup T* destination_,
      ushort simd_group_id,
      ushort simd_lane_id,
      short BLOCK_ROWS_,
      short BLOCK_COLS_,
      short destination_leading_dimension_,
      short reduction_dim,
      short threadgroup_size
  )
      : BLOCK_ROWS(BLOCK_ROWS_), destination_leading_dimension(destination_leading_dimension_),
        vector_size((BLOCK_COLS_ * BLOCK_ROWS_) / threadgroup_size),
        THREAD_ROWS(threadgroup_size / (BLOCK_COLS_ / ((BLOCK_COLS_ * BLOCK_ROWS_) / threadgroup_size))),
        source_leading_dimension(source_leading_dimension_),
        tile_stride(reduction_dim ? BLOCK_COLS_ : BLOCK_ROWS_ * source_leading_dimension_),
        block_row(short(simd_group_id * 32 + simd_lane_id) /
           short(BLOCK_COLS_ / ((BLOCK_COLS_ * BLOCK_ROWS_) / threadgroup_size))),
        block_col(short((BLOCK_COLS_ * BLOCK_ROWS_) / threadgroup_size) *
           (short(simd_group_id * 32 + simd_lane_id) %
            short(BLOCK_COLS_ / ((BLOCK_COLS_ * BLOCK_ROWS_) / threadgroup_size)))),
        destination(destination_ + block_row * destination_leading_dimension_ + block_col),
        source(source_ + block_row * source_leading_dimension_ + block_col) {}

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unchecked() const {
    for (short i = 0; i < BLOCK_ROWS; i += THREAD_ROWS) {
      for (short j = 0; j < vector_size; j++) {
        destination[i * destination_leading_dimension + j] =
            source[i * source_leading_dimension + j];
      }
    }
  }

  /* Load from device memory into threadgroup memory - with bound checking */
  METAL_FUNC void load_checked(short2 source_tile_dimensions) const {
    source_tile_dimensions = source_tile_dimensions - short2(block_col, block_row);

    if (source_tile_dimensions.x <= 0 || source_tile_dimensions.y <= 0) {
      for (short i = 0; i < BLOCK_ROWS; i += THREAD_ROWS) {
        for (short j = 0; j < vector_size; j++) {
          destination[i * destination_leading_dimension + j] = T(0);
        }
      }
      return;
    }

    for (short i = 0; i < BLOCK_ROWS; i += THREAD_ROWS) {
      for (short j = 0; j < vector_size; j++) {
        bool valid = (i < source_tile_dimensions.y) && (j < source_tile_dimensions.x);
        T val = source[(valid ? i * source_leading_dimension + j : 0)];
        destination[i * destination_leading_dimension + j] = valid ? val : T(0);
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() { source += tile_stride; }
};

} // namespace steel
