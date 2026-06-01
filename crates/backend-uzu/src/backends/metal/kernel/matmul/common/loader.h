#pragma once

#include "../../common/defines.h"
#include "../../common/thread_context.h"

using namespace metal;

namespace uzu {
namespace matmul {

///////////////////////////////////////////////////////////////////////////////
// Threadgroup Loader - loads tiles from device memory to threadgroup memory
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    ushort THREADGROUP_TILE_ROWS,
    ushort THREADGROUP_TILE_COLS,
    ushort DESTINATION_LEADING_DIMENSION,
    ushort REDUCTION_DIMENSION,
    ushort THREADGROUP_SIZE,
    ushort ALIGNMENT = 1,
    ushort READS_PER_THREAD = (THREADGROUP_TILE_COLS * THREADGROUP_TILE_ROWS) / (THREADGROUP_SIZE),
    ushort THREAD_COLS = THREADGROUP_TILE_COLS / READS_PER_THREAD,
    ushort THREAD_ROWS = THREADGROUP_SIZE / THREAD_COLS>
struct ThreadgroupLoader {
  const int source_leading_dimension;
  const int tile_stride;

  const ushort thread_index;
  const ushort tile_row_index;
  const ushort tile_col_index;

  threadgroup T* destination;
  const device T* source;

  struct alignas(ALIGNMENT * sizeof(T)) ReadVector {
    uint8_t bytes[sizeof(T) * READS_PER_THREAD];
  };

  METAL_FUNC ThreadgroupLoader(
      const device T* source_ptr,
      const int source_leading_dim,
      threadgroup T* destination_ptr,
      const thread ThreadContext& thread_context
  )
      : source_leading_dimension(source_leading_dim),
        tile_stride(REDUCTION_DIMENSION ? THREADGROUP_TILE_COLS : THREADGROUP_TILE_ROWS * source_leading_dim),
        thread_index(thread_context.simdgroup_index * METAL_SIMD_SIZE + thread_context.simd_lane_id),
        tile_row_index(thread_index / THREAD_COLS), tile_col_index(READS_PER_THREAD * (thread_index % THREAD_COLS)),
        destination(destination_ptr + tile_row_index * DESTINATION_LEADING_DIMENSION + tile_col_index),
        source(source_ptr + tile_row_index * source_leading_dim + tile_col_index) {}

  template <typename UnaryOp>
  METAL_FUNC void apply_inplace_op(thread const UnaryOp& operation) const {
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < THREADGROUP_TILE_ROWS; i += THREAD_ROWS) {
      METAL_PRAGMA_UNROLL
      for (ushort j = 0; j < READS_PER_THREAD; j++) {
        destination[i * DESTINATION_LEADING_DIMENSION + j] =
            operation.apply(destination[i * DESTINATION_LEADING_DIMENSION + j]);
      }
    }
  }

  METAL_FUNC void load_unsafe() const {
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < THREADGROUP_TILE_ROWS; i += THREAD_ROWS) {
      *reinterpret_cast<threadgroup ReadVector*>(&destination[i * DESTINATION_LEADING_DIMENSION]) =
          *reinterpret_cast<const device ReadVector*>(&source[i * source_leading_dimension]);
    }
  }

  METAL_FUNC void load_safe(short2 source_tile_dimensions) const {
    source_tile_dimensions = source_tile_dimensions - short2(tile_col_index, tile_row_index);

    if (source_tile_dimensions.x <= 0 || source_tile_dimensions.y <= 0) {
      METAL_PRAGMA_UNROLL
      for (ushort i = 0; i < THREADGROUP_TILE_ROWS; i += THREAD_ROWS) {
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
    for (ushort i = 0; i < THREADGROUP_TILE_ROWS; i += THREAD_ROWS) {
      METAL_PRAGMA_UNROLL
      for (ushort j = 0; j < READS_PER_THREAD; j++) {
        valid_mask[j] = (i < source_tile_dimensions.y) && (j < source_tile_dimensions.x);
      }

      METAL_PRAGMA_UNROLL
      for (ushort j = 0; j < READS_PER_THREAD; j++) {
        loaded_values[j] = source[(valid_mask[j] ? i * source_leading_dimension + j : 0)];
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
