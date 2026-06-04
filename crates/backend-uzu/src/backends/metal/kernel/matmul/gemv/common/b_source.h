#pragma once

#include "../../common/qdot.h"
#include "../../common/quant_pack.h"
#include "gemv_common.h"
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
    bool INPUT_ALIGNED>
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
        result[0] +=
            dot(static_cast<float4>(*reinterpret_cast<const device W4*>(w)),
                xv);
        result[1] +=
            dot(static_cast<float4>(
                    *reinterpret_cast<const device W4*>(w + in_vec_size)
                ),
                xv);
        result[2] +=
            dot(static_cast<float4>(
                    *reinterpret_cast<const device W4*>(w + 2 * in_vec_size)
                ),
                xv);
        result[3] +=
            dot(static_cast<float4>(
                    *reinterpret_cast<const device W4*>(w + 3 * in_vec_size)
                ),
                xv);
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
          const device BT* w0 = w;
          const device BT* w1 = w + in_vec_size;
          const device BT* w2 = w + 2 * in_vec_size;
          const device BT* w3 = w + 3 * in_vec_size;
          for (int j = 0; j < remaining; j++) {
            U x = static_cast<U>(in[j]);
            result[0] += static_cast<U>(w0[j]) * x;
            result[1] += static_cast<U>(w1[j]) * x;
            result[2] += static_cast<U>(w2[j]) * x;
            result[3] += static_cast<U>(w3[j]) * x;
          }
        }
      }
    } else {
      constexpr uint packs_per_thread = BITS == 2 ? 1 : 2;
      constexpr uint pack_factor = get_pack_factor<BITS, 32>();
      constexpr uint bytes_per_pack = get_bytes_per_pack<BITS, 32>();
      constexpr uint values_per_thread = pack_factor * packs_per_thread;
      constexpr uint block_size = values_per_thread * METAL_SIMD_SIZE;
      constexpr uint scale_step_per_thread = GROUP_SIZE / values_per_thread;
      const device uint8_t* ws = (const device uint8_t*)b;
      thread U x_thread[values_per_thread];

      const uint in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
      const uint in_vec_size_g = (in_vec_size + GROUP_SIZE - 1) / GROUP_SIZE;
      ws += out_row * in_vec_size_w +
            simd_lane * packs_per_thread * bytes_per_pack;

      QuantRowOffsets<BT, U, B_PROLOGUE, BITS> prep;
      prep.group_stride = in_vec_size_g;
      const uint g_offset = simd_lane / scale_step_per_thread;
      prep.scales = scales + out_row * in_vec_size_g + g_offset;
      if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant) {
        prep.biases = biases + out_row * in_vec_size_g + g_offset;
      } else if constexpr (
          B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant
      ) {
        if (BITS == 4) {
          prep.zp_stride = (in_vec_size_g + 1) / 2;
          prep.zps = zero_points + out_row * prep.zp_stride + g_offset / 2;
          prep.high_nibble = (g_offset & 1);
        } else {
          prep.zp_stride = in_vec_size_g;
          prep.zps = zero_points + out_row * prep.zp_stride + g_offset;
        }
      }

      const device AT* in =
          a + batch_idx * in_vec_size + simd_lane * values_per_thread;

      uint k = 0;
      for (; k + block_size <= in_vec_size; k += block_size) {
        U sum = load_vector<AT, U, values_per_thread, BITS>(in, x_thread);

        const device uint8_t* wl0 = ws;
        const device uint8_t* wl1 = ws + in_vec_size_w;
        const device uint8_t* wl2 = ws + 2 * in_vec_size_w;
        const device uint8_t* wl3 = ws + 3 * in_vec_size_w;

        U scale[4];
        U offset[4];
        prep.load(scale, offset);
        result[0] += qdot<U, values_per_thread, BITS>(
            wl0,
            x_thread,
            scale[0],
            offset[0],
            sum
        );
        result[1] += qdot<U, values_per_thread, BITS>(
            wl1,
            x_thread,
            scale[1],
            offset[1],
            sum
        );
        result[2] += qdot<U, values_per_thread, BITS>(
            wl2,
            x_thread,
            scale[2],
            offset[2],
            sum
        );
        result[3] += qdot<U, values_per_thread, BITS>(
            wl3,
            x_thread,
            scale[3],
            offset[3],
            sum
        );

        ws += block_size * bytes_per_pack / pack_factor;
        prep.advance(block_size / GROUP_SIZE);
        in += block_size;
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

          const device uint8_t* wl0 = ws;
          const device uint8_t* wl1 = ws + in_vec_size_w;
          const device uint8_t* wl2 = ws + 2 * in_vec_size_w;
          const device uint8_t* wl3 = ws + 3 * in_vec_size_w;

          U scale[4];
          U offset[4];
          prep.load(scale, offset);
          result[0] += qdot_safe<U, values_per_thread, BITS>(
              wl0,
              x_thread,
              scale[0],
              offset[0],
              sum,
              remaining
          );
          result[1] += qdot_safe<U, values_per_thread, BITS>(
              wl1,
              x_thread,
              scale[1],
              offset[1],
              sum,
              remaining
          );
          result[2] += qdot_safe<U, values_per_thread, BITS>(
              wl2,
              x_thread,
              scale[2],
              offset[2],
              sum,
              remaining
          );
          result[3] += qdot_safe<U, values_per_thread, BITS>(
              wl3,
              x_thread,
              scale[3],
              offset[3],
              sum,
              remaining
          );
        }
      }
    }
  }
};

} // namespace gemm
} // namespace uzu
