#pragma once

#include <metal_simdgroup>
#include <metal_stdlib>

#include "../../common/defines.h"
#include "quant_pack.h"
#include "quant_unpack.h"

using namespace metal;

namespace uzu {
namespace gemm {

// Block loader for `QuantizationMethod::ScaleZeroPoint` weights (per-group
// scale, per-group integer zero point; bias is derived as `-scale * zp`).
//
// `per_output_layout = false` (the default) expects the zero-point byte stream
// laid out per output row: each row has `ceil(groups_per_row / 2)` bytes for
// bits=4 (2 zp's per byte) or `groups_per_row` bytes for bits=8.
//
// `per_output_layout = true` switches to a (K-row, group-col) layout used by
// some embedding paths — left in place to preserve the legacy QMM behavior.
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits,
    bool per_output_layout = false>
struct QuantizedBlockLoaderScaleZeroPoint {
  static_assert(
      BCOLS <= group_size,
      "Group size should be larger than columns"
  );
  static_assert(
      group_size % BCOLS == 0,
      "Group size should be divisible by columns"
  );
  static_assert(bits == 4 || bits == 8, "Only int4 and int8 supported");

  METAL_CONST short pack_factor = get_pack_factor<bits, 8>();
  METAL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  METAL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  METAL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  METAL_CONST short group_steps = group_size / BCOLS;

  const int src_ld;
  const int groups_per_row;
  const int tile_stride;
  short group_step_cnt;
  int k_base;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  const device T* scales;
  const device T* scales_row_start;
  const device uint8_t* zps_row_start;
  const int out_group_base;
  const int out_groups_total;
  const int zp_stride_total;

  QuantizedBlockLoaderScaleZeroPoint(
      const device uint8_t* src_,
      const device T* scales_,
      const device uint8_t* zero_points_row_start_,
      const int src_ld_,
      const int groups_per_row_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      const int out_group_base_ = 0,
      const int out_groups_total_ = 0,
      const int zp_stride_total_ = 0
  )
      : src_ld(src_ld_), groups_per_row(groups_per_row_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED * bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor
        ),
        group_step_cnt(0), k_base(0), group_stride(BROWS * groups_per_row_),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(reduction_dim == 1 ? (scales_ + bi * groups_per_row_) : scales_),
        scales_row_start(
            reduction_dim == 1 ? (scales_ + bi * groups_per_row_) : scales_
        ),
        zps_row_start(
            reduction_dim == 1 ? (zero_points_row_start_ +
                                  bi * (bits == 4 ? ((groups_per_row_ + 1) / 2)
                                                  : groups_per_row_))
                               : zero_points_row_start_
        ),
        out_group_base(per_output_layout ? out_group_base_ : 0),
        out_groups_total(per_output_layout ? out_groups_total_ : 0),
        zp_stride_total(per_output_layout ? zp_stride_total_ : 0) {}

  inline void current_scale_bias(
      thread T& out_scale,
      thread T& out_bias
  ) const {
    uint zp_n;
    T scale_val;
    if (per_output_layout) {
      const int row_idx = k_base + bi;
      const int scale_index = row_idx * groups_per_row + out_group_base;
      scale_val = scales_row_start[scale_index];
      if (bits == 4) {
        const int byte_index =
            row_idx * zp_stride_total + (out_group_base >> 1);
        uint8_t zp_b = zps_row_start[byte_index];
        zp_n = (uint(zp_b) >> (uint(out_group_base & 1) * 4u)) & 0x0Fu;
      } else {
        const int zp_index = row_idx * zp_stride_total + out_group_base;
        zp_n = zps_row_start[zp_index];
      }
    } else {
      int g = reduction_dim == 0 ? (k_base / group_size)
                                 : (int)(scales - scales_row_start);
      scale_val = reduction_dim == 0 ? scales_row_start[g] : *scales;
      if (bits == 4) {
        const device uint8_t* zp_ptr = zps_row_start + (g >> 1);
        uint8_t zp_b = *zp_ptr;
        zp_n = (uint(zp_b) >> (uint(g & 1) * 4u)) & 0x0Fu;
      } else {
        zp_n = zps_row_start[g];
      }
    }
    out_scale = scale_val;
    out_bias = static_cast<T>(-scale_val * static_cast<T>(zp_n));
  }

  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    T scale;
    T bias;
    current_scale_bias(scale, bias);
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          src + i * bytes_per_pack,
          scale,
          bias,
          dst + i * pack_factor
      );
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (reduction_dim == 1) {
      // N-tail: zero out rows beyond valid outputs
      if (bi >= src_tile_dim.x) {
        for (int i = 0; i < n_reads * pack_factor; i++) {
          dst[i] = T(0);
        }
        return;
      }

      int valid_cols = src_tile_dim.y; // 0..BK
      int valid_packs = (valid_cols + pack_factor - 1) / pack_factor;

      T scale;
      T bias;
      current_scale_bias(scale, bias);
      for (int i = 0; i < n_reads; i++) {
        int pack_idx = bj + i; // global pack index across the BK packs
        if (pack_idx < valid_packs) {
          dequantize<T, pack_factor, bits>(
              src + i * bytes_per_pack,
              scale,
              bias,
              dst + i * pack_factor
          );

          // Mask the last pack if needed
          if (pack_idx == valid_packs - 1) {
            int rem = valid_cols - pack_idx * pack_factor;
            if (rem < pack_factor) {
              for (int r = rem; r < pack_factor; ++r) {
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

    if (reduction_dim == 0 && bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    T scale;
    T bias;
    current_scale_bias(scale, bias);
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          src + i * bytes_per_pack,
          scale,
          bias,
          dst + i * pack_factor
      );
    }
  }

  void next() {
    src += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_cnt++;
        if (group_step_cnt == group_steps) {
          group_step_cnt = 0;
          scales++;
        }
      } else {
        scales++;
      }
    } else {
      k_base += BROWS;
    }
  }
};

} // namespace gemm
} // namespace uzu
