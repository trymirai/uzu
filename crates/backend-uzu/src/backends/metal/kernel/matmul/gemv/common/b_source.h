#pragma once

#include "../../common/qdot.h"
#include "../../common/quant_pack.h"
#include "quant_row_offsets.h"

namespace uzu {
namespace gemm {

template <
    typename BT,
    typename AT,
    typename U,
    GemmBPrologueKind B_PROLOGUE,
    uint GROUP_SIZE,
    uint BITS,
    uint K_SPLIT,
    bool INPUT_ALIGNED,
    uint RESULTS_PER_SIMDGROUP,
    uint PACKS_PER_THREAD>
struct BSource {
  static METAL_FUNC void accumulate(
      thread U (&result)[RESULTS_PER_SIMDGROUP],
      const device uint32_t* b,
      const device BT* scales,
      const device uint8_t* zero_points,
      const device BT* biases,
      const device AT* a,
      uint in_vec_size,
      uint out_row,
      uint batch_idx,
      uint simd_lane,
      uint k_slice
  ) {
    if constexpr (B_PROLOGUE == GemmBPrologueKind::FullPrecision) {
      constexpr uint values_per_thread = 4;
      constexpr uint block_size = values_per_thread * METAL_SIMD_SIZE;
      typedef vec<BT, 4> W4;
      typedef vec<AT, 4> I4;
      const uint k_stride = K_SPLIT * block_size;
      const uint k_start = k_slice * block_size;
      const device BT* w = reinterpret_cast<const device BT*>(b);
      w += out_row * in_vec_size + simd_lane * values_per_thread + k_start;
      const device AT* in =
          a + batch_idx * in_vec_size + simd_lane * values_per_thread + k_start;

      uint k = k_start;
      for (; k + block_size <= in_vec_size; k += k_stride) {
        float4 xv =
            static_cast<float4>(*reinterpret_cast<const device I4*>(in));
        METAL_PRAGMA_UNROLL
        for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
          result[row] += dot(
              static_cast<float4>(
                  *reinterpret_cast<const device W4*>(w + row * in_vec_size)
              ),
              xv
          );
        }
        w += k_stride;
        in += k_stride;
      }

      if constexpr (K_SPLIT == 1 && !INPUT_ALIGNED) {
        const uint thread_offset = simd_lane * values_per_thread;
        const int remaining =
            (k + thread_offset < in_vec_size)
                ? min(static_cast<int>(in_vec_size - k - thread_offset),
                      static_cast<int>(values_per_thread))
                : 0;
        if (remaining > 0) {
          for (int j = 0; j < remaining; j++) {
            U x = static_cast<U>(in[j]);
            METAL_PRAGMA_UNROLL
            for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
              result[row] += static_cast<U>(w[row * in_vec_size + j]) * x;
            }
          }
        }
      }
    } else {
      constexpr uint packs_per_thread = PACKS_PER_THREAD;
      constexpr uint pack_factor = get_pack_factor<BITS, 32>();
      constexpr uint bytes_per_pack = get_bytes_per_pack<BITS, 32>();
      constexpr uint values_per_thread = pack_factor * packs_per_thread;
      constexpr uint block_size = values_per_thread * METAL_SIMD_SIZE;
      constexpr uint k_stride = K_SPLIT * block_size;
      constexpr uint scale_step_per_thread = GROUP_SIZE / values_per_thread;
      const device uint8_t* ws = (const device uint8_t*)b;
      thread U x_thread[values_per_thread];

      const uint in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
      const uint in_vec_size_g = (in_vec_size + GROUP_SIZE - 1) / GROUP_SIZE;
      const uint k_start = k_slice * block_size;
      const uint k_start_w = k_start * bytes_per_pack / pack_factor;
      const uint g_start = k_start / GROUP_SIZE;
      ws += out_row * in_vec_size_w +
            simd_lane * packs_per_thread * bytes_per_pack + k_start_w;

      QuantRowOffsets<BT, U, B_PROLOGUE, BITS> prep;
      prep.group_stride = in_vec_size_g;
      const uint g_offset = simd_lane / scale_step_per_thread;
      prep.scales = scales + out_row * in_vec_size_g + g_start + g_offset;
      if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant) {
        prep.biases = biases + out_row * in_vec_size_g + g_start + g_offset;
      } else if constexpr (
          B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant
      ) {
        if (BITS == 4) {
          prep.zp_stride = (in_vec_size_g + 1) / 2;
          prep.zps = zero_points + out_row * prep.zp_stride +
                     (g_start + g_offset) / 2;
          prep.high_nibble = ((g_start + g_offset) & 1);
        } else {
          prep.zp_stride = in_vec_size_g;
          prep.zps = zero_points + out_row * prep.zp_stride + g_start +
                     g_offset;
        }
      }

      const device AT* in =
          a + batch_idx * in_vec_size + simd_lane * values_per_thread +
          k_start;

      uint k = k_start;
      for (; k + block_size <= in_vec_size; k += k_stride) {
        U sum = load_vector<AT, U, values_per_thread, BITS>(in, x_thread);

        U scale[RESULTS_PER_SIMDGROUP];
        U offset[RESULTS_PER_SIMDGROUP];
        prep.load(scale, offset);
        METAL_PRAGMA_UNROLL
        for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
          result[row] += qdot<U, values_per_thread, BITS>(
              ws + row * in_vec_size_w,
              x_thread,
              scale[row],
              offset[row],
              sum
          );
        }

        ws += k_stride * bytes_per_pack / pack_factor;
        prep.advance(k_stride / GROUP_SIZE);
        in += k_stride;
      }

      if constexpr (!INPUT_ALIGNED) {
        const uint thread_offset = simd_lane * values_per_thread;
        const int remaining =
            (k + thread_offset < in_vec_size)
                ? min(static_cast<int>(in_vec_size - k - thread_offset),
                      static_cast<int>(values_per_thread))
                : 0;
        if (remaining > 0) {
          U sum = load_vector_safe<AT, U, values_per_thread>(
              in,
              x_thread,
              remaining
          );

          U scale[RESULTS_PER_SIMDGROUP];
          U offset[RESULTS_PER_SIMDGROUP];
          prep.load(scale, offset);
          METAL_PRAGMA_UNROLL
          for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
            result[row] += qdot_safe<U, values_per_thread, BITS>(
                ws + row * in_vec_size_w,
                x_thread,
                scale[row],
                offset[row],
                sum,
                remaining
            );
          }
        }
      }
    }
  }
};

} // namespace gemm
} // namespace uzu
