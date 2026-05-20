#pragma once

#include <metal_simdgroup>
#include <metal_stdlib>

#include "../../common/defines.h"
#include "quant_pack.h"
#include "quant_unpack.h"

using namespace metal;

namespace uzu {
namespace gemm {

template <
    typename T,
    short THREADGROUP_TILE_ROWS,
    short THREADGROUP_TILE_COLS,
    short DESTINATION_LEADING_DIMENSION,
    short REDUCTION_DIMENSION,
    short THREADGROUP_SIZE,
    short GROUP_SIZE,
    short BITS,
    bool PER_OUTPUT_LAYOUT = false>
struct QuantizedBlockLoaderScaleZeroPoint {
  static_assert(
      THREADGROUP_TILE_COLS <= GROUP_SIZE,
      "Group size should be larger than columns"
  );
  static_assert(
      GROUP_SIZE % THREADGROUP_TILE_COLS == 0,
      "Group size should be divisible by columns"
  );
  static_assert(BITS == 4 || BITS == 8, "Only int4 and int8 supported");

  METAL_CONST short pack_factor = get_pack_factor<BITS, 8>();
  METAL_CONST short bytes_per_pack = get_bytes_per_pack<BITS>();
  METAL_CONST short THREADGROUP_TILE_COLS_PACKED = THREADGROUP_TILE_COLS / pack_factor;
  METAL_CONST short READS_PER_THREAD =
      (THREADGROUP_TILE_COLS_PACKED * THREADGROUP_TILE_ROWS < THREADGROUP_SIZE)
          ? 1
          : (THREADGROUP_TILE_COLS_PACKED * THREADGROUP_TILE_ROWS) / THREADGROUP_SIZE;
  METAL_CONST short GROUP_STEPS_PER_BLOCK = GROUP_SIZE / THREADGROUP_TILE_COLS;

  const int src_leading_dim;
  const int groups_per_row;
  const int tile_stride;
  short group_step_counter;
  int k_base;
  const int group_stride;

  const short thread_index;
  const short tile_row_index;
  const short tile_col_index;

  threadgroup T* dst;
  const device uint8_t* src;
  const device T* scales;
  const device T* scales_row_start;
  const device uint8_t* zero_points_row_start;
  const int output_group_base;
  const int output_groups_total;
  const int zero_point_stride_total;

  QuantizedBlockLoaderScaleZeroPoint(
      const device uint8_t* src_,
      const device T* scales_,
      const device uint8_t* zero_points_row_start_,
      const int src_leading_dim_,
      const int groups_per_row_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      const int output_group_base_ = 0,
      const int output_groups_total_ = 0,
      const int zero_point_stride_total_ = 0
  )
      : src_leading_dim(src_leading_dim_), groups_per_row(groups_per_row_),
        tile_stride(
            REDUCTION_DIMENSION
                ? THREADGROUP_TILE_COLS_PACKED * bytes_per_pack
                : THREADGROUP_TILE_ROWS * src_leading_dim_ * bytes_per_pack / pack_factor
        ),
        group_step_counter(0), k_base(0),
        group_stride(THREADGROUP_TILE_ROWS * groups_per_row_),
        thread_index(simd_group_id * 32 + simd_lane_id),
        tile_row_index(READS_PER_THREAD * thread_index / THREADGROUP_TILE_COLS_PACKED),
        tile_col_index((READS_PER_THREAD * thread_index) % THREADGROUP_TILE_COLS_PACKED),
        dst(dst_ + tile_row_index * DESTINATION_LEADING_DIMENSION +
            tile_col_index * pack_factor),
        src(src_ +
            tile_row_index * src_leading_dim_ * bytes_per_pack / pack_factor +
            tile_col_index * bytes_per_pack),
        scales(
            REDUCTION_DIMENSION == 1 ? (scales_ + tile_row_index * groups_per_row_)
                               : scales_
        ),
        scales_row_start(
            REDUCTION_DIMENSION == 1 ? (scales_ + tile_row_index * groups_per_row_)
                               : scales_
        ),
        zero_points_row_start(
            REDUCTION_DIMENSION == 1
                ? (zero_points_row_start_ +
                   tile_row_index * (BITS == 4 ? ((groups_per_row_ + 1) / 2)
                                                : groups_per_row_))
                : zero_points_row_start_
        ),
        output_group_base(PER_OUTPUT_LAYOUT ? output_group_base_ : 0),
        output_groups_total(PER_OUTPUT_LAYOUT ? output_groups_total_ : 0),
        zero_point_stride_total(
            PER_OUTPUT_LAYOUT ? zero_point_stride_total_ : 0
        ) {}

  inline void current_scale_bias(
      thread T& out_scale,
      thread T& out_bias
  ) const {
    uint zero_point_value;
    T scale_value;
    if (PER_OUTPUT_LAYOUT) {
      const int row_index = k_base + tile_row_index;
      const int scale_index =
          row_index * groups_per_row + output_group_base;
      scale_value = scales_row_start[scale_index];
      if (BITS == 4) {
        const int byte_index =
            row_index * zero_point_stride_total + (output_group_base >> 1);
        uint8_t zero_point_byte = zero_points_row_start[byte_index];
        zero_point_value =
            (uint(zero_point_byte) >> (uint(output_group_base & 1) * 4u)) &
            0x0Fu;
      } else {
        const int zero_point_index =
            row_index * zero_point_stride_total + output_group_base;
        zero_point_value = zero_points_row_start[zero_point_index];
      }
    } else {
      int group_index = REDUCTION_DIMENSION == 0
          ? (k_base / GROUP_SIZE)
          : static_cast<int>(scales - scales_row_start);
      scale_value = REDUCTION_DIMENSION == 0 ? scales_row_start[group_index]
                                       : *scales;
      if (BITS == 4) {
        const device uint8_t* zero_point_ptr =
            zero_points_row_start + (group_index >> 1);
        uint8_t zero_point_byte = *zero_point_ptr;
        zero_point_value =
            (uint(zero_point_byte) >> (uint(group_index & 1) * 4u)) & 0x0Fu;
      } else {
        zero_point_value = zero_points_row_start[group_index];
      }
    }
    out_scale = scale_value;
    out_bias =
        static_cast<T>(-scale_value * static_cast<T>(zero_point_value));
  }

  void load_unsafe() const {
    if (THREADGROUP_TILE_COLS_PACKED * THREADGROUP_TILE_ROWS < THREADGROUP_SIZE &&
        tile_row_index >= THREADGROUP_TILE_ROWS) {
      return;
    }

    T scale;
    T bias;
    current_scale_bias(scale, bias);
    for (int i = 0; i < READS_PER_THREAD; i++) {
      dequantize<T, pack_factor, BITS>(
          src + i * bytes_per_pack,
          scale,
          bias,
          dst + i * pack_factor
      );
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (THREADGROUP_TILE_COLS_PACKED * THREADGROUP_TILE_ROWS < THREADGROUP_SIZE &&
        tile_row_index >= THREADGROUP_TILE_ROWS) {
      return;
    }

    if (REDUCTION_DIMENSION == 1) {
      if (tile_row_index >= src_tile_dim.x) {
        for (int i = 0; i < READS_PER_THREAD * pack_factor; i++) {
          dst[i] = T(0);
        }
        return;
      }

      int valid_cols = src_tile_dim.y;
      int valid_packs = (valid_cols + pack_factor - 1) / pack_factor;

      T scale;
      T bias;
      current_scale_bias(scale, bias);
      for (int i = 0; i < READS_PER_THREAD; i++) {
        int pack_index = tile_col_index + i;
        if (pack_index < valid_packs) {
          dequantize<T, pack_factor, BITS>(
              src + i * bytes_per_pack,
              scale,
              bias,
              dst + i * pack_factor
          );

          if (pack_index == valid_packs - 1) {
            int remaining = valid_cols - pack_index * pack_factor;
            if (remaining < pack_factor) {
              for (int r = remaining; r < pack_factor; ++r) {
                dst[i * pack_factor + r] = T(0);
              }
            }
          }
        } else {
          for (int j = 0; j < pack_factor; ++j) {
            dst[i * pack_factor + j] = T(0);
          }
        }
      }
      return;
    }

    if (REDUCTION_DIMENSION == 0 && tile_row_index >= src_tile_dim.y) {
      for (int i = 0; i < READS_PER_THREAD * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    T scale;
    T bias;
    current_scale_bias(scale, bias);
    for (int i = 0; i < READS_PER_THREAD; i++) {
      dequantize<T, pack_factor, BITS>(
          src + i * bytes_per_pack,
          scale,
          bias,
          dst + i * pack_factor
      );
    }
  }

  void next() {
    src += tile_stride;
    if (REDUCTION_DIMENSION == 1) {
      if (GROUP_STEPS_PER_BLOCK > 1) {
        group_step_counter++;
        if (group_step_counter == GROUP_STEPS_PER_BLOCK) {
          group_step_counter = 0;
          scales++;
        }
      } else {
        scales++;
      }
    } else {
      k_base += THREADGROUP_TILE_ROWS;
    }
  }
};

} // namespace gemm
} // namespace uzu
