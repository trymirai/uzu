#pragma once

#include <metal_simdgroup>
#include <metal_stdlib>

#include "../../common/defines.h"
#include "../../common/quant_pack.h"
#include "../../common/quant_unpack.h"

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
    bool SCALE_SYMMETRIC = false>
struct QuantizedBlockLoaderScaleZeroPoint {
  static_assert(THREADGROUP_TILE_COLS <= GROUP_SIZE, "Group size should be larger than columns");
  static_assert(GROUP_SIZE % THREADGROUP_TILE_COLS == 0, "Group size should be divisible by columns");
  static_assert(BITS == 4 || BITS == 8, "Only int4 and int8 supported");

  UZU_CONST short PACK_FACTOR = get_pack_factor<BITS, 8>();
  UZU_CONST short BYTES_PER_PACK = get_bytes_per_pack<BITS>();
  UZU_CONST short THREADGROUP_TILE_COLS_PACKED = THREADGROUP_TILE_COLS / PACK_FACTOR;
  UZU_CONST short READS_PER_THREAD = (THREADGROUP_TILE_COLS_PACKED * THREADGROUP_TILE_ROWS < THREADGROUP_SIZE)
                                         ? 1
                                         : (THREADGROUP_TILE_COLS_PACKED * THREADGROUP_TILE_ROWS) / THREADGROUP_SIZE;
  UZU_CONST short GROUP_STEPS_PER_BLOCK = GROUP_SIZE / THREADGROUP_TILE_COLS;
  UZU_CONST bool TILE_HAS_IDLE_THREADS = THREADGROUP_TILE_COLS_PACKED * THREADGROUP_TILE_ROWS < THREADGROUP_SIZE;

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
  const bool signed_codes;

  QuantizedBlockLoaderScaleZeroPoint(
      const device uint8_t* src_,
      const device T* scales_,
      const device uint8_t* zero_points_row_start_,
      const bool signed_codes_,
      const int src_leading_dim_,
      const int groups_per_row_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]
  )
      : src_leading_dim(src_leading_dim_), groups_per_row(groups_per_row_),
        tile_stride(
            REDUCTION_DIMENSION ? THREADGROUP_TILE_COLS_PACKED * BYTES_PER_PACK
                                : THREADGROUP_TILE_ROWS * src_leading_dim_ * BYTES_PER_PACK / PACK_FACTOR
        ),
        group_step_counter(0), k_base(0), group_stride(THREADGROUP_TILE_ROWS * groups_per_row_),
        thread_index(simd_group_id * 32 + simd_lane_id),
        tile_row_index(READS_PER_THREAD * thread_index / THREADGROUP_TILE_COLS_PACKED),
        tile_col_index((READS_PER_THREAD * thread_index) % THREADGROUP_TILE_COLS_PACKED),
        dst(dst_ + tile_row_index * DESTINATION_LEADING_DIMENSION + tile_col_index * PACK_FACTOR),
        src(src_ + tile_row_index * src_leading_dim_ * BYTES_PER_PACK / PACK_FACTOR + tile_col_index * BYTES_PER_PACK),
        scales(REDUCTION_DIMENSION == 1 ? (scales_ + tile_row_index * groups_per_row_) : scales_),
        scales_row_start(REDUCTION_DIMENSION == 1 ? (scales_ + tile_row_index * groups_per_row_) : scales_),
        zero_points_row_start(
            SCALE_SYMMETRIC
                ? nullptr
                : (REDUCTION_DIMENSION == 1 ? (zero_points_row_start_ +
                                               tile_row_index * zero_point_row_stride<ushort(BITS)>(groups_per_row_))
                                            : zero_points_row_start_)
        ),
        signed_codes(signed_codes_) {}

  QuantizedBlockLoaderScaleZeroPoint(
      const device uint8_t* src_,
      const device T* scales_,
      const bool signed_codes_,
      const int src_leading_dim_,
      const int groups_per_row_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]
  )
      : QuantizedBlockLoaderScaleZeroPoint(
            src_,
            scales_,
            static_cast<const device uint8_t*>(nullptr),
            signed_codes_,
            src_leading_dim_,
            groups_per_row_,
            dst_,
            simd_group_id,
            simd_lane_id
        ) {
    static_assert(SCALE_SYMMETRIC, "zero-point-free loader construction requires symmetric quantization");
  }

  inline void current_scale_bias(thread T& out_scale, thread T& out_bias) const {
    uint zero_point_value;
    T scale_value;
    int group_index;
    if constexpr (REDUCTION_DIMENSION == 0) {
      group_index = k_base / GROUP_SIZE;
      scale_value = scales_row_start[group_index];
    } else {
      group_index = static_cast<int>(scales - scales_row_start);
      scale_value = *scales;
    }
    if constexpr (SCALE_SYMMETRIC) {
      zero_point_value = symmetric_zero_point<ushort(BITS)>();
    } else {
      zero_point_value = decode_zero_point<ushort(BITS)>(zero_points_row_start, uint(group_index));
    }
    out_scale = scale_value;
    out_bias = static_cast<T>(-scale_value * static_cast<T>(zero_point_value));
  }

  void load_unsafe() const {
    if constexpr (TILE_HAS_IDLE_THREADS) {
      if (tile_row_index >= THREADGROUP_TILE_ROWS) {
        return;
      }
    }

    T scale;
    T bias;
    current_scale_bias(scale, bias);
    for (int i = 0; i < READS_PER_THREAD; i++) {
      dequantize<T, PACK_FACTOR, BITS>(src + i * BYTES_PER_PACK, scale, bias, dst + i * PACK_FACTOR, signed_codes);
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if constexpr (TILE_HAS_IDLE_THREADS) {
      if (tile_row_index >= THREADGROUP_TILE_ROWS) {
        return;
      }
    }

    if (tile_row_index >= src_tile_dim.y) {
      for (int i = 0; i < READS_PER_THREAD * PACK_FACTOR; i++) {
        dst[i] = T(0);
      }
      return;
    }

    const int valid_cols = src_tile_dim.x;
    const int valid_packs = (valid_cols + PACK_FACTOR - 1) / PACK_FACTOR;
    T scale;
    T bias;
    current_scale_bias(scale, bias);
    for (int i = 0; i < READS_PER_THREAD; i++) {
      const int pack_index = tile_col_index + i;
      if (pack_index < valid_packs) {
        dequantize<T, PACK_FACTOR, BITS>(src + i * BYTES_PER_PACK, scale, bias, dst + i * PACK_FACTOR, signed_codes);
        if (pack_index == valid_packs - 1) {
          const int remaining = valid_cols - pack_index * PACK_FACTOR;
          for (int lane = remaining; lane < PACK_FACTOR; ++lane) {
            dst[i * PACK_FACTOR + lane] = T(0);
          }
        }
      } else {
        for (int lane = 0; lane < PACK_FACTOR; ++lane) {
          dst[i * PACK_FACTOR + lane] = T(0);
        }
      }
    }
  }

  void next() {
    src += tile_stride;
    if constexpr (REDUCTION_DIMENSION == 1) {
      if constexpr (GROUP_STEPS_PER_BLOCK > 1) {
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
