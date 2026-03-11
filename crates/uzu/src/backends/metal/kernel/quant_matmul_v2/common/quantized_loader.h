#pragma once

#include "dequantize.h"
#include "../../matmul/common/mma.h"
#include "../../matmul/common/loader.h"

namespace uzu {
namespace quantized_matmul {

template <
    typename T,
    short BLOCK_ROWS,
    short BLOCK_COLS,
    short destination_leading_dimension,
    short reduction_dim,
    short threadgroup_size,
    short group_size,
    short bits>
struct QuantizedBlockLoaderAffineBias {
  static_assert(
      BLOCK_COLS <= group_size,
      "Group size should be larger than columns"
  );
  static_assert(
      group_size % BLOCK_COLS == 0,
      "Group size should be divisible by columns"
  );
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits, 8>();
  MTL_CONST short BLOCK_COLS_PACKED = BLOCK_COLS / pack_factor;
  MTL_CONST short read_count =
      (BLOCK_COLS_PACKED * BLOCK_ROWS < threadgroup_size)
          ? 1
          : (BLOCK_COLS_PACKED * BLOCK_ROWS) / threadgroup_size;
  MTL_CONST short group_steps = group_size / BLOCK_COLS;

  const int source_leading_dimension;
  const int tile_stride;
  short group_step_counter;
  const int group_stride;

  const short thread_idx;
  const short block_row;
  const short block_col;

  threadgroup T* destination;
  const device uint8_t* source;
  const device T* scales;
  const device T* biases;

  QuantizedBlockLoaderAffineBias(
      const device uint8_t* source_,
      const device T* scales_,
      const device T* biases_,
      const int source_leading_dimension_,
      threadgroup T* destination_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]
  )
      : source_leading_dimension(source_leading_dimension_),
        tile_stride(
            reduction_dim
                ? BLOCK_COLS_PACKED * bytes_per_pack
                : BLOCK_ROWS * source_leading_dimension * bytes_per_pack /
                      pack_factor
        ),
        group_step_counter(0),
        group_stride(BLOCK_ROWS * source_leading_dimension / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        block_row(read_count * thread_idx / BLOCK_COLS_PACKED),
        block_col((read_count * thread_idx) % BLOCK_COLS_PACKED),
        destination(
            destination_ + block_row * destination_leading_dimension +
            block_col * pack_factor
        ),
        source(
            source_ +
            block_row * source_leading_dimension * bytes_per_pack /
                pack_factor +
            block_col * bytes_per_pack
        ),
        scales(scales_ + block_row * source_leading_dimension / group_size),
        biases(biases_ + block_row * source_leading_dimension / group_size) {}

  void load_unchecked() const {
    if (BLOCK_COLS_PACKED * BLOCK_ROWS < threadgroup_size &&
        block_row >= BLOCK_ROWS) {
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < read_count; i++) {
      dequantize<T, pack_factor, bits>(
          source + i * bytes_per_pack,
          scale,
          bias,
          destination + i * pack_factor
      );
    }
  }

  void load_checked(short2 source_tile_dimensions) const {
    if (BLOCK_COLS_PACKED * BLOCK_ROWS < threadgroup_size &&
        block_row >= BLOCK_ROWS) {
      return;
    }

    if (reduction_dim == 1 && block_row >= source_tile_dimensions.x) {
      for (int i = 0; i < read_count * pack_factor; i++) {
        destination[i] = T(0);
      }
      return;
    }

    if (reduction_dim == 0 && block_row >= source_tile_dimensions.y) {
      for (int i = 0; i < read_count * pack_factor; i++) {
        destination[i] = T(0);
      }
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < read_count; i++) {
      dequantize<T, pack_factor, bits>(
          (device uint8_t*)(source + i * bytes_per_pack),
          scale,
          bias,
          destination + i * pack_factor
      );
    }
  }

  void next() {
    source += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_counter++;
        if (group_step_counter == group_steps) {
          group_step_counter = 0;
          scales++;
          biases++;
        }
      } else {
        scales++;
        biases++;
      }
    } else {
      scales += group_stride;
      biases += group_stride;
    }
  }
};

template <
    typename T,
    short BLOCK_ROWS,
    short BLOCK_COLS,
    short destination_leading_dimension,
    short reduction_dim,
    short threadgroup_size,
    short group_size,
    short bits,
    bool per_output_layout = false>
struct QuantizedBlockLoaderZeroPoint {
  static_assert(
      BLOCK_COLS <= group_size,
      "Group size should be larger than columns"
  );
  static_assert(
      group_size % BLOCK_COLS == 0,
      "Group size should be divisible by columns"
  );
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits, 8>();
  MTL_CONST short BLOCK_COLS_PACKED = BLOCK_COLS / pack_factor;
  MTL_CONST short read_count =
      (BLOCK_COLS_PACKED * BLOCK_ROWS < threadgroup_size)
          ? 1
          : (BLOCK_COLS_PACKED * BLOCK_ROWS) / threadgroup_size;
  MTL_CONST short group_steps = group_size / BLOCK_COLS;

  const int source_leading_dimension;
  const int groups_per_row;
  const int tile_stride;
  short group_step_counter;
  int k_base;
  const int group_stride;

  const short thread_idx;
  const short block_row;
  const short block_col;

  threadgroup T* destination;
  const device uint8_t* source;
  const device T* scales;
  const device T* scales_row_start;
  const device uint8_t* zero_points_row_start;
  const int output_group_base;
  const int output_groups_total;
  const int zero_point_stride_total;

  QuantizedBlockLoaderZeroPoint(
      const device uint8_t* source_,
      const device T* scales_,
      const device uint8_t* zero_points_row_start_,
      const int source_leading_dimension_,
      const int groups_per_row_,
      threadgroup T* destination_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      const int output_group_base_ = 0,
      const int output_groups_total_ = 0,
      const int zero_point_stride_total_ = 0
  )
      : source_leading_dimension(source_leading_dimension_),
        groups_per_row(groups_per_row_),
        tile_stride(
            reduction_dim
                ? BLOCK_COLS_PACKED * bytes_per_pack
                : BLOCK_ROWS * source_leading_dimension * bytes_per_pack /
                      pack_factor
        ),
        group_step_counter(0), k_base(0),
        group_stride(BLOCK_ROWS * groups_per_row_),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        block_row(read_count * thread_idx / BLOCK_COLS_PACKED),
        block_col((read_count * thread_idx) % BLOCK_COLS_PACKED),
        destination(
            destination_ + block_row * destination_leading_dimension +
            block_col * pack_factor
        ),
        source(
            source_ +
            block_row * source_leading_dimension * bytes_per_pack /
                pack_factor +
            block_col * bytes_per_pack
        ),
        scales(
            reduction_dim == 1 ? (scales_ + block_row * groups_per_row_)
                               : scales_
        ),
        scales_row_start(
            reduction_dim == 1 ? (scales_ + block_row * groups_per_row_)
                               : scales_
        ),
        zero_points_row_start(
            reduction_dim == 1
                ? (zero_points_row_start_ +
                   block_row *
                       (bits == 4 ? ((groups_per_row_ + 1) / 2)
                                  : groups_per_row_))
                : zero_points_row_start_
        ),
        output_group_base(per_output_layout ? output_group_base_ : 0),
        output_groups_total(per_output_layout ? output_groups_total_ : 0),
        zero_point_stride_total(
            per_output_layout ? zero_point_stride_total_ : 0
        ) {}

  inline void current_scale_and_bias(
      thread T& out_scale,
      thread T& out_bias
  ) const {
    uint zero_point_numeric;
    T scale_value;
    if (per_output_layout) {
      const int row_index = k_base + block_row;
      const int scale_index =
          row_index * groups_per_row + output_group_base;
      scale_value = scales_row_start[scale_index];
      if (bits == 4) {
        const int byte_index =
            row_index * zero_point_stride_total + (output_group_base >> 1);
        uint8_t zero_point_byte = zero_points_row_start[byte_index];
        zero_point_numeric = (output_group_base & 1)
                                 ? ((zero_point_byte >> 4) & 0x0F)
                                 : (zero_point_byte & 0x0F);
      } else {
        const int zero_point_index =
            row_index * zero_point_stride_total + output_group_base;
        zero_point_numeric = zero_points_row_start[zero_point_index];
      }
    } else {
      int group_index = reduction_dim == 0
                            ? (k_base / group_size)
                            : (int)(scales - scales_row_start);
      scale_value = reduction_dim == 0 ? scales_row_start[group_index]
                                       : *scales;
      if (bits == 4) {
        const device uint8_t* zero_point_pointer =
            zero_points_row_start + (group_index >> 1);
        uint8_t zero_point_byte = *zero_point_pointer;
        zero_point_numeric = (group_index & 1)
                                 ? ((zero_point_byte >> 4) & 0x0F)
                                 : (zero_point_byte & 0x0F);
      } else {
        zero_point_numeric = zero_points_row_start[group_index];
      }
    }
    out_scale = scale_value;
    out_bias =
        static_cast<T>(-scale_value * static_cast<T>(zero_point_numeric));
  }

  void load_unchecked() const {
    if (BLOCK_COLS_PACKED * BLOCK_ROWS < threadgroup_size &&
        block_row >= BLOCK_ROWS) {
      return;
    }

    T scale;
    T bias;
    current_scale_and_bias(scale, bias);
    for (int i = 0; i < read_count; i++) {
      dequantize<T, pack_factor, bits>(
          source + i * bytes_per_pack,
          scale,
          bias,
          destination + i * pack_factor
      );
    }
  }

  void load_checked(short2 source_tile_dimensions) const {
    if (BLOCK_COLS_PACKED * BLOCK_ROWS < threadgroup_size &&
        block_row >= BLOCK_ROWS) {
      return;
    }

    if (reduction_dim == 1) {
      if (block_row >= source_tile_dimensions.x) {
        for (int i = 0; i < read_count * pack_factor; i++) {
          destination[i] = T(0);
        }
        return;
      }

      int valid_columns = source_tile_dimensions.y;
      int valid_packs = (valid_columns + pack_factor - 1) / pack_factor;

      T scale;
      T bias;
      current_scale_and_bias(scale, bias);
      for (int i = 0; i < read_count; i++) {
        int pack_index = block_col + i;
        if (pack_index < valid_packs) {
          dequantize<T, pack_factor, bits>(
              source + i * bytes_per_pack,
              scale,
              bias,
              destination + i * pack_factor
          );

          if (pack_index == valid_packs - 1) {
            int remainder = valid_columns - pack_index * pack_factor;
            if (remainder < pack_factor) {
              for (int r = remainder; r < pack_factor; ++r) {
                destination[i * pack_factor + r] = T(0);
              }
            }
          }
        } else {
          for (int j = 0; j < pack_factor; ++j) {
            destination[i * pack_factor + j] = T(0);
          }
        }
      }
      return;
    }

    if (reduction_dim == 0 && block_row >= source_tile_dimensions.y) {
      for (int i = 0; i < read_count * pack_factor; i++) {
        destination[i] = T(0);
      }
      return;
    }

    T scale;
    T bias;
    current_scale_and_bias(scale, bias);
    for (int i = 0; i < read_count; i++) {
      dequantize<T, pack_factor, bits>(
          source + i * bytes_per_pack,
          scale,
          bias,
          destination + i * pack_factor
      );
    }
  }

  void next() {
    source += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_counter++;
        if (group_step_counter == group_steps) {
          group_step_counter = 0;
          scales++;
        }
      } else {
        scales++;
      }
    } else {
      k_base += BLOCK_ROWS;
    }
  }
};

} // namespace quantized_matmul
} // namespace uzu
