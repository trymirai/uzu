#pragma once

#include <metal_simdgroup>
#include <metal_stdlib>

#include "../../common/defines.h"
#include "../../common/quant_pack.h"
#include "../../layout/interleaved_rows_layout.h"
#include "../../common/quant_unpack.h"

using namespace metal;

namespace uzu {
namespace gemm {

template <
    typename T,
    short THREADGROUP_TILE_ROWS,
    short THREADGROUP_TILE_COLS,
    short DESTINATION_LEADING_DIMENSION,
    short THREADGROUP_SIZE,
    short GROUP_SIZE,
    short BITS,
    typename StageT,
    typename ComputeT>
struct QuantizedBlockLoaderScaleBias {
  static_assert(THREADGROUP_TILE_COLS <= GROUP_SIZE, "Group size should be larger than columns");
  static_assert(GROUP_SIZE % THREADGROUP_TILE_COLS == 0, "Group size should be divisible by columns");
  static_assert(BITS == 2 || BITS == 4 || BITS == 8, "Only int2, int4 and int8 supported");

  UZU_CONST short PACK_FACTOR = get_pack_factor<BITS, 8>();
  UZU_CONST short BYTES_PER_PACK = get_bytes_per_pack<BITS>();
  UZU_CONST short THREADGROUP_TILE_COLS_PACKED = THREADGROUP_TILE_COLS / PACK_FACTOR;
  UZU_CONST short READS_PER_THREAD = (THREADGROUP_TILE_COLS_PACKED * THREADGROUP_TILE_ROWS < THREADGROUP_SIZE)
                                         ? 1
                                         : (THREADGROUP_TILE_COLS_PACKED * THREADGROUP_TILE_ROWS) / THREADGROUP_SIZE;
  UZU_CONST short GROUP_STEPS_PER_BLOCK = GROUP_SIZE / THREADGROUP_TILE_COLS;
  UZU_CONST bool TILE_HAS_IDLE_THREADS = THREADGROUP_TILE_COLS_PACKED * THREADGROUP_TILE_ROWS < THREADGROUP_SIZE;
  static_assert(
      DESTINATION_LEADING_DIMENSION % 2 == 0 && (READS_PER_THREAD * PACK_FACTOR) % 2 == 0,
      "paired stores need 4-byte alignment"
  );

  const int src_leading_dim;
  const int tile_stride;
  short group_step_counter;
  const int group_stride;

  const short thread_index;
  const short tile_row_index;
  const short tile_col_index;

  threadgroup StageT* dst;
  const device uint8_t* src;
  const device T* scales;
  const device T* biases;
  const bool signed_codes;
  const bool interleaved_rows;

  static METAL_FUNC int source_offset(bool interleaved_rows, int byte) {
    return interleaved_rows ? InterleavedRowsLayout::byte_offset(byte) : byte;
  }

  QuantizedBlockLoaderScaleBias(
      const device uint8_t* src_,
      const device T* scales_,
      const device T* biases_,
      const bool signed_codes_,
      const bool interleaved_rows_,
      const int src_leading_dim_,
      const int params_group_stride_,
      const int params_output_stride_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]
  )
      : src_leading_dim(src_leading_dim_),
        tile_stride(source_offset(interleaved_rows_, THREADGROUP_TILE_COLS_PACKED * BYTES_PER_PACK)),
        group_step_counter(0), group_stride(params_group_stride_), thread_index(simd_group_id * 32 + simd_lane_id),
        tile_row_index(READS_PER_THREAD * thread_index / THREADGROUP_TILE_COLS_PACKED),
        tile_col_index((READS_PER_THREAD * thread_index) % THREADGROUP_TILE_COLS_PACKED),
        dst(reinterpret_cast<threadgroup StageT*>(dst_) + tile_row_index * DESTINATION_LEADING_DIMENSION +
            tile_col_index * PACK_FACTOR),
        src(src_ +
            (interleaved_rows_
                 ? sizeof(uint) *
                       InterleavedRowsLayout::word_index(tile_row_index, 0, src_leading_dim_ / W4_CODES_PER_WORD)
                 : tile_row_index * src_leading_dim_ * BYTES_PER_PACK / PACK_FACTOR) +
            source_offset(interleaved_rows_, tile_col_index * BYTES_PER_PACK)),
        scales(scales_ + tile_row_index * params_output_stride_),
        biases(biases_ + tile_row_index * params_output_stride_), signed_codes(signed_codes_),
        interleaved_rows(interleaved_rows_) {}

  void load_unsafe() const {
    if constexpr (TILE_HAS_IDLE_THREADS) {
      if (tile_row_index >= THREADGROUP_TILE_ROWS) {
        return;
      }
    }

    dequantize<StageT, ComputeT, READS_PER_THREAD * PACK_FACTOR, BITS>(
        src,
        float(*scales),
        float(*biases),
        dst,
        signed_codes,
        interleaved_rows ? InterleavedRowsLayout::STRIP_ROWS : 1
    );
  }

  void load_safe(short2 src_tile_dim) const {
    if constexpr (TILE_HAS_IDLE_THREADS) {
      if (tile_row_index >= THREADGROUP_TILE_ROWS) {
        return;
      }
    }

    if (tile_row_index >= src_tile_dim.y) {
      for (int i = 0; i < READS_PER_THREAD * PACK_FACTOR; i++) {
        dst[i] = StageT(0);
      }
      return;
    }

    const int valid_cols = src_tile_dim.x;
    const int valid_packs = (valid_cols + PACK_FACTOR - 1) / PACK_FACTOR;
    const float scale = float(*scales);
    const float bias = float(*biases);
    for (int i = 0; i < READS_PER_THREAD; i++) {
      const int pack_index = tile_col_index + i;
      if (pack_index < valid_packs) {
        dequantize<StageT, ComputeT, PACK_FACTOR, BITS>(
            src + source_offset(interleaved_rows, i * BYTES_PER_PACK),
            scale,
            bias,
            dst + i * PACK_FACTOR,
            signed_codes
        );
        if (pack_index == valid_packs - 1) {
          const int remaining = valid_cols - pack_index * PACK_FACTOR;
          for (int lane = remaining; lane < PACK_FACTOR; ++lane) {
            dst[i * PACK_FACTOR + lane] = StageT(0);
          }
        }
      } else {
        for (int lane = 0; lane < PACK_FACTOR; ++lane) {
          dst[i * PACK_FACTOR + lane] = StageT(0);
        }
      }
    }
  }

  void next() {
    src += tile_stride;
    if constexpr (GROUP_STEPS_PER_BLOCK > 1) {
      group_step_counter++;
      if (group_step_counter != GROUP_STEPS_PER_BLOCK) {
        return;
      }
      group_step_counter = 0;
    }
    scales += group_stride;
    biases += group_stride;
  }
};

} // namespace gemm
} // namespace uzu
