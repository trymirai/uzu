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
    short BITS>
struct QuantizedBlockLoaderScaleBias {
  static_assert(THREADGROUP_TILE_COLS <= GROUP_SIZE, "Group size should be larger than columns");
  static_assert(GROUP_SIZE % THREADGROUP_TILE_COLS == 0, "Group size should be divisible by columns");
  static_assert(BITS == 4 || BITS == 8, "Only int4 and int8 supported");

  METAL_CONST short pack_factor = get_pack_factor<BITS, 8>();
  METAL_CONST short bytes_per_pack = get_bytes_per_pack<BITS>();
  METAL_CONST short THREADGROUP_TILE_COLS_PACKED = THREADGROUP_TILE_COLS / pack_factor;
  METAL_CONST short READS_PER_THREAD = (THREADGROUP_TILE_COLS_PACKED * THREADGROUP_TILE_ROWS < THREADGROUP_SIZE)
                                           ? 1
                                           : (THREADGROUP_TILE_COLS_PACKED * THREADGROUP_TILE_ROWS) / THREADGROUP_SIZE;
  METAL_CONST short GROUP_STEPS_PER_BLOCK = GROUP_SIZE / THREADGROUP_TILE_COLS;
  METAL_CONST bool TILE_HAS_IDLE_THREADS = THREADGROUP_TILE_COLS_PACKED * THREADGROUP_TILE_ROWS < THREADGROUP_SIZE;

  const int src_leading_dim;
  const int tile_stride;
  short group_step_counter;
  const int group_stride;

  const short thread_index;
  const short tile_row_index;
  const short tile_col_index;

  threadgroup T* dst;
  const device uint8_t* src;
  const device T* scales;
  const device T* biases;

  QuantizedBlockLoaderScaleBias(
      const device uint8_t* src_,
      const device T* scales_,
      const device T* biases_,
      const int src_leading_dim_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]
  )
      : src_leading_dim(src_leading_dim_),
        tile_stride(
            REDUCTION_DIMENSION ? THREADGROUP_TILE_COLS_PACKED * bytes_per_pack
                                : THREADGROUP_TILE_ROWS * src_leading_dim_ * bytes_per_pack / pack_factor
        ),
        group_step_counter(0), group_stride(THREADGROUP_TILE_ROWS * src_leading_dim_ / GROUP_SIZE),
        thread_index(simd_group_id * 32 + simd_lane_id),
        tile_row_index(READS_PER_THREAD * thread_index / THREADGROUP_TILE_COLS_PACKED),
        tile_col_index((READS_PER_THREAD * thread_index) % THREADGROUP_TILE_COLS_PACKED),
        dst(dst_ + tile_row_index * DESTINATION_LEADING_DIMENSION + tile_col_index * pack_factor),
        src(src_ + tile_row_index * src_leading_dim_ * bytes_per_pack / pack_factor + tile_col_index * bytes_per_pack),
        scales(scales_ + tile_row_index * src_leading_dim_ / GROUP_SIZE),
        biases(biases_ + tile_row_index * src_leading_dim_ / GROUP_SIZE) {}

  void load_unsafe() const {
    if constexpr (TILE_HAS_IDLE_THREADS) {
      if (tile_row_index >= THREADGROUP_TILE_ROWS) {
        return;
      }
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < READS_PER_THREAD; i++) {
      dequantize<T, pack_factor, BITS>(src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if constexpr (TILE_HAS_IDLE_THREADS) {
      if (tile_row_index >= THREADGROUP_TILE_ROWS) {
        return;
      }
    }

    if constexpr (REDUCTION_DIMENSION == 1) {
      if (tile_row_index >= src_tile_dim.x) {
        for (int i = 0; i < READS_PER_THREAD * pack_factor; i++) {
          dst[i] = T(0);
        }
        return;
      }
    } else {
      if (tile_row_index >= src_tile_dim.y) {
        for (int i = 0; i < READS_PER_THREAD * pack_factor; i++) {
          dst[i] = T(0);
        }
        return;
      }
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < READS_PER_THREAD; i++) {
      dequantize<T, pack_factor, BITS>(src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
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

} // namespace gemm
} // namespace uzu
