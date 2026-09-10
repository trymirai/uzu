#include <metal_stdlib>

#include "../common/defines.h"
#include "../common/dsl.h"

using namespace metal;

#define TILE_SIZE 32
#define TILE_ROW_STRIDE (TILE_SIZE + 1)
#define THREADS_PER_THREADGROUP 128

struct TileBounds {
  uint row_start;
  uint column_start;
  uint row_count;
  uint column_count;

  METAL_FUNC TileBounds transposed() const { return {column_start, row_start, column_count, row_count}; }
};

template <ushort BITS>
METAL_FUNC void read_elements(
    const device uchar* source,
    uint source_row_stride,
    TileBounds bounds,
    uint tile_row,
    uint read_in_row,
    threadgroup ushort* tile
) {
  constexpr uint elements_per_read = BITS == 4 ? 2 : 1;
  constexpr uint bytes_per_read = BITS == 16 ? 2 : 1;
  const uint tile_column = read_in_row * elements_per_read;
  const uint byte_index = (bounds.row_start + tile_row) * source_row_stride +
                          (bounds.column_start / elements_per_read + read_in_row) * bytes_per_read;

  if constexpr (BITS == 4) {
    const uchar packed = source[byte_index];
    tile[tile_row * TILE_ROW_STRIDE + tile_column] = ushort(packed & 0xf);
    if (tile_column + 1 < bounds.column_count) {
      tile[tile_row * TILE_ROW_STRIDE + tile_column + 1] = ushort(packed >> 4);
    }
  } else if constexpr (BITS == 8) {
    tile[tile_row * TILE_ROW_STRIDE + tile_column] = ushort(source[byte_index]);
  } else {
    tile[tile_row * TILE_ROW_STRIDE + tile_column] = ushort(source[byte_index]) | (ushort(source[byte_index + 1]) << 8);
  }
}

template <ushort BITS>
METAL_FUNC void load_tile(
    const device uchar* source,
    uint source_row_stride,
    TileBounds bounds,
    threadgroup ushort* tile,
    uint thread_index
) {
  constexpr uint elements_per_read = BITS == 4 ? 2 : 1;
  const uint reads_per_row = (bounds.column_count + elements_per_read - 1) / elements_per_read;
  const uint read_count = bounds.row_count * reads_per_row;

  for (uint read_index = thread_index; read_index < read_count; read_index += THREADS_PER_THREADGROUP) {
    const uint tile_row = read_index / reads_per_row;
    const uint read_in_row = read_index % reads_per_row;
    read_elements<BITS>(source, source_row_stride, bounds, tile_row, read_in_row, tile);
  }
}

template <ushort BITS>
METAL_FUNC void write_elements(
    device uchar* output,
    uint output_row_stride,
    TileBounds bounds,
    uint write_row,
    uint write_in_row,
    threadgroup const ushort* tile
) {
  constexpr uint elements_per_write = BITS == 4 ? 2 : 1;
  constexpr uint bytes_per_write = BITS == 16 ? 2 : 1;
  const uint source_row = write_in_row * elements_per_write;
  const uint source_column = write_row;
  const uint byte_index = (bounds.row_start + write_row) * output_row_stride +
                          (bounds.column_start / elements_per_write + write_in_row) * bytes_per_write;

  if constexpr (BITS == 4) {
    uchar packed = uchar(tile[source_row * TILE_ROW_STRIDE + source_column] & 0xf);
    if (source_row + 1 < bounds.column_count) {
      packed |= uchar(tile[(source_row + 1) * TILE_ROW_STRIDE + source_column] & 0xf) << 4;
    }
    output[byte_index] = packed;
  } else if constexpr (BITS == 8) {
    output[byte_index] = uchar(tile[source_row * TILE_ROW_STRIDE + source_column]);
  } else {
    const ushort value = tile[source_row * TILE_ROW_STRIDE + source_column];
    output[byte_index] = uchar(value);
    output[byte_index + 1] = uchar(value >> 8);
  }
}

template <ushort BITS>
METAL_FUNC void store_tile(
    device uchar* output,
    uint output_row_stride,
    TileBounds bounds,
    threadgroup const ushort* tile,
    uint thread_index
) {
  constexpr uint elements_per_write = BITS == 4 ? 2 : 1;
  const uint writes_per_row = (bounds.column_count + elements_per_write - 1) / elements_per_write;
  const uint write_count = bounds.row_count * writes_per_row;

  for (uint write_index = thread_index; write_index < write_count; write_index += THREADS_PER_THREADGROUP) {
    const uint write_row = write_index / writes_per_row;
    const uint write_in_row = write_index % writes_per_row;
    write_elements<BITS>(output, output_row_stride, bounds, write_row, write_in_row, tile);
  }
}

template <ushort BITS>
VARIANTS(BITS, 4, 8, 16)
PUBLIC KERNEL(Transpose)(
    device uchar* input,
    device uchar* output OPTIONAL(!in_place),
    constant uint& rows,
    constant uint& cols,
    const bool in_place SPECIALIZE,
    threadgroup ushort source_tile[TILE_SIZE * TILE_ROW_STRIDE],
    threadgroup ushort opposite_tile[TILE_SIZE * TILE_ROW_STRIDE],
    const uint tile_column_index GROUPS(cols.div_ceil(TILE_SIZE)),
    const uint tile_row_index GROUPS(rows.div_ceil(TILE_SIZE)),
    const uint thread_index THREADS(THREADS_PER_THREADGROUP)
) {
  if (in_place && tile_row_index > tile_column_index) {
    return;
  }

  const uint source_row_stride = (cols * BITS + 7) / 8;
  const uint output_row_stride = (rows * BITS + 7) / 8;
  const device uchar* source = input;
  device uchar* destination = in_place ? input : output;
  const uint row_start = tile_row_index * TILE_SIZE;
  const uint column_start = tile_column_index * TILE_SIZE;
  const TileBounds source_bounds = {
      row_start,
      column_start,
      min(rows - row_start, uint(TILE_SIZE)),
      min(cols - column_start, uint(TILE_SIZE)),
  };
  const TileBounds destination_bounds = source_bounds.transposed();
  const bool off_diagonal_in_place = in_place && tile_row_index != tile_column_index;

  load_tile<BITS>(source, source_row_stride, source_bounds, source_tile, thread_index);
  if (off_diagonal_in_place) {
    load_tile<BITS>(source, source_row_stride, destination_bounds, opposite_tile, thread_index);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  store_tile<BITS>(destination, output_row_stride, destination_bounds, source_tile, thread_index);
  if (off_diagonal_in_place) {
    store_tile<BITS>(destination, output_row_stride, source_bounds, opposite_tile, thread_index);
  }
}
