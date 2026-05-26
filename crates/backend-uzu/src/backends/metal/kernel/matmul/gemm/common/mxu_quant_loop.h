#pragma once

#include "../../../common/thread_context.h"
#include "../../common/fragment.h"
#include "../../common/mxu_fragment_ops.h"
#include "../generated/gemm.h"
#include "quant_pack.h"

using namespace metal;

namespace uzu {
namespace gemm {

// MXU K-loop for quantized B (TRANSPOSE_B = true required).
//
// Mirrors the FP `uzu::matmul::gemm_loop` but populates the right (B)
// Fragment per-thread via inline dequantization of packed weights,
// avoiding any threadgroup-memory roundtrip.
//
// Storage layout (TRANSPOSE_B = true):
//   weights: [N, K] packed at BITS bits per element, row-major
//     - row stride = K * BITS / 8 bytes
//     - element address: weights + n*row_stride + (k*BITS/8) bytes,
//       with a nibble select on BITS=4
//   scales/biases: [N, num_groups_k] in T, with num_groups_k = K / GROUP_SIZE
//   zero_points (ScaleZeroPoint, BITS=4): [N, (num_groups_k+1)/2] bytes
//   zero_points (ScaleZeroPoint, BITS=8): [N, num_groups_k] bytes
template <
    typename T,
    int SIMDGROUP_BLOCK_M,
    int SIMDGROUP_BLOCK_N,
    int SIMDGROUP_BLOCK_K,
    int THREADGROUP_BLOCK_K,
    bool aligned_m,
    bool aligned_n,
    GemmBPrologueKind B_PROLOGUE,
    int BITS,
    int GROUP_SIZE,
    typename AccumulatorType = float>
METAL_FUNC auto mxu_quant_gemm_loop(
    const device T* a_ptr,
    const device uint8_t* weights,
    const device T* scales,
    const device T* biases,
    const device uint8_t* zero_points,
    int leading_dimension_a,
    int K,
    int n_simdgroup_base,
    int aligned_k_iterations,
    const short simdgroup_limit_m,
    const short simdgroup_limit_n,
    const thread ThreadContext& thread_context
) {
  using FragOps = uzu::matmul::MxuFragmentOps;
  constexpr ushort TILES_M = SIMDGROUP_BLOCK_M / FragOps::FRAGMENT_ROWS;
  constexpr ushort TILES_N = SIMDGROUP_BLOCK_N / FragOps::FRAGMENT_COLS;
  constexpr ushort TILES_K = SIMDGROUP_BLOCK_K / FragOps::FRAGMENT_ROWS;

  // TRANSPOSE_B = true layout: B fragment is [N, K] sub-tile.
  constexpr ushort LEFT_TILE_ROWS = TILES_M;
  constexpr ushort LEFT_TILE_COLS = TILES_K;
  constexpr ushort RIGHT_TILE_ROWS = TILES_N;
  constexpr ushort RIGHT_TILE_COLS = TILES_K;

  uzu::matmul::Fragment<
      AccumulatorType,
      TILES_M,
      TILES_N,
      FragOps>
      accumulator(thread_context);
  accumulator.clear();

  // Thread's per-fragment (row, col) origin within a 16x16 sub-tile.
  using BFragType = uzu::matmul::
      Fragment<T, RIGHT_TILE_ROWS, RIGHT_TILE_COLS, FragOps>;
  const short2 pos = BFragType::get_position(thread_context);

  constexpr int pack_factor = get_pack_factor<BITS, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<BITS, 8>();
  const int row_stride_bytes = K * bytes_per_pack / pack_factor;
  const int num_groups_k = K / GROUP_SIZE;
  const int zp_stride_bytes = (BITS == 4) ? ((num_groups_k + 1) / 2) : num_groups_k;

  METAL_PRAGMA_NO_UNROLL
  for (int outer_k = 0; outer_k < aligned_k_iterations; ++outer_k) {
    threadgroup_barrier(mem_flags::mem_none);

    METAL_PRAGMA_NO_UNROLL
    for (int inner_k = 0; inner_k < THREADGROUP_BLOCK_K;
         inner_k += SIMDGROUP_BLOCK_K) {
      const int k_block_base = outer_k * THREADGROUP_BLOCK_K + inner_k;

      // --- Load A (FP) ---
      uzu::matmul::Fragment<T, LEFT_TILE_ROWS, LEFT_TILE_COLS, FragOps>
          left_tile(thread_context);
      const int left_offset = inner_k;
      if constexpr (aligned_m) {
        left_tile.load(a_ptr + left_offset, leading_dimension_a);
      } else {
        left_tile.load_safe(
            a_ptr + left_offset,
            leading_dimension_a,
            short2(SIMDGROUP_BLOCK_K, simdgroup_limit_m));
      }

      // --- Build right_tile per-thread via inline dequant ---
      uzu::matmul::Fragment<T, RIGHT_TILE_ROWS, RIGHT_TILE_COLS, FragOps>
          right_tile(thread_context);

      METAL_PRAGMA_UNROLL
      for (ushort tile_row = 0; tile_row < RIGHT_TILE_ROWS; ++tile_row) {
        METAL_PRAGMA_UNROLL
        for (ushort tile_col = 0; tile_col < RIGHT_TILE_COLS; ++tile_col) {
          thread auto& frag = right_tile.fragment_at(tile_row, tile_col);
          METAL_PRAGMA_UNROLL
          for (ushort i = 0; i < FragOps::THREAD_ELEMENT_ROWS; ++i) {
            const int n_in_frag = int(tile_row) * int(FragOps::FRAGMENT_ROWS) +
                int(pos.y) + int(i) * int(FragOps::THREAD_ELEMENT_ROW_STRIDE);
            const int n_global = n_simdgroup_base + n_in_frag;
            const bool n_in_bounds = aligned_n || (n_in_frag < int(simdgroup_limit_n));
            METAL_PRAGMA_UNROLL
            for (ushort j = 0; j < FragOps::THREAD_ELEMENT_COLS; ++j) {
              const int k_in_frag = int(tile_col) * int(FragOps::FRAGMENT_COLS) +
                  int(pos.x) + int(j);
              const int k_global = k_block_base + k_in_frag;

              T value = T(0);
              if (n_in_bounds) {
                // Read packed weight raw value.
                uint8_t raw;
                if constexpr (BITS == 4) {
                  const int byte_idx = n_global * row_stride_bytes + (k_global >> 1);
                  const uint8_t byte = weights[byte_idx];
                  raw = (k_global & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
                } else {
                  const int byte_idx = n_global * row_stride_bytes + k_global;
                  raw = weights[byte_idx];
                }

                const int group = k_global / GROUP_SIZE;
                const T scale = scales[n_global * num_groups_k + group];
                T bias_val;
                if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant) {
                  bias_val = biases[n_global * num_groups_k + group];
                } else {
                  uint8_t zp;
                  if constexpr (BITS == 4) {
                    const uint8_t zp_byte =
                        zero_points[n_global * zp_stride_bytes + (group >> 1)];
                    zp = (group & 1) ? ((zp_byte >> 4) & 0x0F) : (zp_byte & 0x0F);
                  } else {
                    zp = zero_points[n_global * zp_stride_bytes + group];
                  }
                  bias_val = -scale * T(zp);
                }
                value = scale * T(raw) + bias_val;
              }

              const ushort element_index = i * FragOps::THREAD_ELEMENT_COLS + j;
              frag[element_index] = value;
            }
          }
        }
      }

      FragOps::template tile_matmul<false, true>(
          accumulator, left_tile, right_tile);
    }
  }

  return accumulator;
}

} // namespace gemm
} // namespace uzu
