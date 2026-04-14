#pragma once

#include "../../common/defines.h"

using namespace metal;

namespace uzu {
namespace matmul {

///////////////////////////////////////////////////////////////////////////////
// Threadgroup Loader - loads tiles from device memory to threadgroup memory
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    ushort BLOCK_ROWS,
    ushort BLOCK_COLS,
    ushort DESTINATION_LEADING_DIMENSION,
    ushort REDUCTION_DIMENSION,
    ushort THREADGROUP_SIZE,
    ushort ALIGNMENT = 1,
    ushort READS_PER_THREAD = (BLOCK_COLS * BLOCK_ROWS) / (THREADGROUP_SIZE),
    ushort THREAD_COLS = BLOCK_COLS / READS_PER_THREAD,
    ushort THREAD_ROWS = THREADGROUP_SIZE / THREAD_COLS>
struct ThreadgroupLoader {
  METAL_CONST ushort ROW_ITERATIONS =
      (BLOCK_ROWS + THREAD_ROWS - 1) / THREAD_ROWS;

  const int source_leading_dimension;
  const int tile_stride;

  const ushort thread_index;
  const ushort block_row_index;
  const ushort block_col_index;

  threadgroup T* destination;
  const device T* source;

  struct alignas(ALIGNMENT * sizeof(T)) ReadVector {
    uint8_t bytes[sizeof(T) * READS_PER_THREAD];
  };

  METAL_FUNC ThreadgroupLoader(
      const device T* source_ptr,
      const int source_leading_dim,
      threadgroup T* destination_ptr,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]
  )
      : source_leading_dimension(source_leading_dim),
        tile_stride(
            REDUCTION_DIMENSION ? BLOCK_COLS : BLOCK_ROWS * source_leading_dim
        ),
        thread_index(simd_group_id * METAL_SIMD_SIZE + simd_lane_id),
        block_row_index(thread_index / THREAD_COLS),
        block_col_index(READS_PER_THREAD * (thread_index % THREAD_COLS)),
        destination(
            destination_ptr + block_row_index * DESTINATION_LEADING_DIMENSION +
            block_col_index
        ),
        source(
            source_ptr + block_row_index * source_leading_dim + block_col_index
        ) {}

  template <typename UnaryOp>
  METAL_FUNC void apply_inplace_op(thread const UnaryOp& operation) const {
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < BLOCK_ROWS; i += THREAD_ROWS) {
      METAL_PRAGMA_UNROLL
      for (ushort j = 0; j < READS_PER_THREAD; j++) {
        destination[i * DESTINATION_LEADING_DIMENSION + j] =
            operation.apply(destination[i * DESTINATION_LEADING_DIMENSION + j]);
      }
    }
  }

  METAL_FUNC void load_unsafe() const {
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < BLOCK_ROWS; i += THREAD_ROWS) {
      *((threadgroup ReadVector*)(&destination
                                      [i * DESTINATION_LEADING_DIMENSION])) =
          *((const device ReadVector*)(&source[i * source_leading_dimension]));
    }
  }

  METAL_FUNC void load_safe(short2 source_tile_dimensions) const {
    source_tile_dimensions =
        source_tile_dimensions - short2(block_col_index, block_row_index);

    if (source_tile_dimensions.x <= 0 || source_tile_dimensions.y <= 0) {
      METAL_PRAGMA_UNROLL
      for (ushort i = 0; i < BLOCK_ROWS; i += THREAD_ROWS) {
        METAL_PRAGMA_UNROLL
        for (ushort j = 0; j < READS_PER_THREAD; j++) {
          destination[i * DESTINATION_LEADING_DIMENSION + j] = T(0);
        }
      }
      return;
    }

    bool valid_mask[READS_PER_THREAD];
    T loaded_values[READS_PER_THREAD];

    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < BLOCK_ROWS; i += THREAD_ROWS) {
      METAL_PRAGMA_UNROLL
      for (ushort j = 0; j < READS_PER_THREAD; j++) {
        valid_mask[j] =
            (i < source_tile_dimensions.y) && (j < source_tile_dimensions.x);
      }

      METAL_PRAGMA_UNROLL
      for (ushort j = 0; j < READS_PER_THREAD; j++) {
        loaded_values[j] =
            source[(valid_mask[j] ? i * source_leading_dimension + j : 0)];
      }

      METAL_PRAGMA_UNROLL
      for (ushort j = 0; j < READS_PER_THREAD; j++) {
        loaded_values[j] = valid_mask[j] ? loaded_values[j] : T(0);
      }

      METAL_PRAGMA_UNROLL
      for (ushort j = 0; j < READS_PER_THREAD; j++) {
        destination[i * DESTINATION_LEADING_DIMENSION + j] = loaded_values[j];
      }
    }
  }

  METAL_FUNC void next() { source += tile_stride; }
};

} // namespace matmul
} // namespace uzu
