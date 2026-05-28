#pragma once

#include <metal_stdlib>
#include <metal_simdgroup>

#include "../../../common/defines.h"
#include "../../../common/thread_context.h"
#include "../../../hadamard_transform/hadamard_transform.h"
#include "../../../generated/quantization_method.h"
#include "../../../generated/gemm.h"
#include "../../../generated/matmul.h"
#include "../../common/qdot.h"
#include "../../common/quant_pack.h"
#include "gemv_tiling.h"

using namespace metal;
using namespace uzu::quantization_method;
using namespace uzu::gemm;
using namespace uzu::matmul;

// Upper bound for the full-precision threadgroup reduction scratch (in floats).
// Max config: tg_simd_cols=16, output_rows_per_tg=16, thread_out_rows=4
// → 16 * (16 + 4) = 320. Unused when tg_simd_cols == 1.
#define GEMV_MAX_THREADGROUP_MEMORY 320

// 0-safe wrappers: BITS == 0 is the in-band marker for the full-precision path,
// so we must not instantiate the quant pack helpers with a zero bit-width.
template <uint B>
inline constexpr uint qmv_pack_factor() {
  if constexpr (B == 0) {
    return 1u;
  } else {
    return get_pack_factor<B, 32>();
  }
}
template <uint B>
inline constexpr uint qmv_bytes_per_pack() {
  if constexpr (B == 0) {
    return 0u;
  } else {
    return get_bytes_per_pack<B, 32>();
  }
}

namespace uzu {
namespace gemv {

// Unified GEMV core. The two paths (full-precision vs group-quantized) live in
// the same `run()` body and are selected at compile time via `B_PROLOGUE`.
//
// Both paths agree on:
//   - grid axes: `threadgroup_position.x` = output block on N,
//                `threadgroup_position.y` = batch row on M.
//   - threadgroup geometry: 32 lanes × 8 simdgroups × 1 (set in gemv.metal).
//   - epilogue ordering: optional scale → optional accumulate → optional bias.
//
// They diverge inside the K loop: the FP path streams typed loads through
// `simd_shuffle_down` (with optional cross-simdgroup partial sums via
// `partial_shared`) and is tile-driven; the quantized path bulk-loads packed
// bits and calls `qdot` / `qdot_safe`, with a `simd_sum` finalize and lane-0
// staging via `result_shared`. Tile-aware quant is intentionally deferred —
// the host pins `GemvTiling::Wide` for quant, which matches the constants
// hardcoded in the quant branch below.
template <
    typename T,
    GemmBPrologueKind B_PROLOGUE = GemmBPrologueKind::FullPrecision,
    int BITS = 0,
    int GROUP_SIZE = 0>
struct GemvCore {
  static METAL_FUNC void run(
      const device T* a,
      const device uint8_t* b_packed,
      device T* d,
      const device T* scales,
      const device T* biases,
      const device uint8_t* zero_points,
      const device T* output_bias,
      const device int32_t* rht_factors,
      const constant uzu::matmul::GemvParams* params,
      GemmDTransform output_transform,
      GemvTiling gemv_tiling,
      threadgroup float* partial_shared,
      threadgroup float* result_shared,
      const thread ThreadContext& thread_context
  ) {
    const bool is_scale = output_transform.contains(GemmDTransform::SCALE);
    const bool is_accumulate =
        output_transform.contains(GemmDTransform::ACCUMULATE);
    const bool is_bias = output_transform.contains(GemmDTransform::BIAS);
    const bool use_hadamard = output_transform.contains(GemmDTransform::RHT);

    if constexpr (B_PROLOGUE == GemmBPrologueKind::FullPrecision) {
      (void)scales;
      (void)biases;
      (void)zero_points;
      (void)result_shared;

      const device T* matrix = reinterpret_cast<const device T*>(b_packed);

      if (params->output_rows_per_threadgroup <= 0) {
        return;
      }

      // Kill excess simdgroups beyond what this tile uses.
      if (thread_context.simdgroup_index >=
          gemv_tiling_tg_simd_rows(gemv_tiling) *
              gemv_tiling_tg_simd_cols(gemv_tiling)) {
        return;
      }

      const uint threads_per_threadgroup_col =
          gemv_tiling_tg_simd_cols(gemv_tiling) *
          gemv_tiling_sg_thread_cols(gemv_tiling);
      const uint input_columns_per_threadgroup =
          threads_per_threadgroup_col *
          gemv_tiling_thread_out_cols(gemv_tiling);

      const uint batch_row = thread_context.threadgroup_position.y;
      if (batch_row >= params->batch_size) {
        return;
      }

      thread float accumulated_values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      thread T matrix_values[4];
      thread float vector_coefficients[4];

      const uint thread_row_in_simdgroup =
          gemv_tiling_sg_thread_cols(gemv_tiling) != 32
              ? static_cast<uint>(thread_context.simd_lane_id) /
                  static_cast<uint>(gemv_tiling_sg_thread_cols(gemv_tiling))
              : 0;
      const int thread_col_in_simdgroup =
          gemv_tiling_sg_thread_cols(gemv_tiling) != 32
              ? static_cast<uint>(thread_context.simd_lane_id) %
                  static_cast<uint>(gemv_tiling_sg_thread_cols(gemv_tiling))
              : static_cast<uint>(thread_context.simd_lane_id);

      const int simdgroup_column_index =
          gemv_tiling_tg_simd_cols(gemv_tiling) != 1
              ? static_cast<uint>(
                    thread_context.simdgroup_index %
                    gemv_tiling_tg_simd_cols(gemv_tiling))
              : 0;

      const uint simdgroup_row_thread_base =
          gemv_tiling_tg_simd_cols(gemv_tiling) != 1
              ? static_cast<uint>(gemv_tiling_sg_thread_rows(gemv_tiling)) *
                  static_cast<uint>(
                      thread_context.simdgroup_index /
                      gemv_tiling_tg_simd_cols(gemv_tiling))
              : static_cast<uint>(gemv_tiling_sg_thread_rows(gemv_tiling)) *
                  static_cast<uint>(thread_context.simdgroup_index);
      const uint simdgroup_col_thread_base =
          gemv_tiling_tg_simd_cols(gemv_tiling) != 1
              ? static_cast<uint>(gemv_tiling_sg_thread_cols(gemv_tiling)) *
                  static_cast<uint>(
                      thread_context.simdgroup_index %
                      gemv_tiling_tg_simd_cols(gemv_tiling))
              : 0;

      uint output_block_row_offset =
          (simdgroup_row_thread_base + thread_row_in_simdgroup) *
          static_cast<uint>(gemv_tiling_thread_out_rows(gemv_tiling));
      uint input_block_col_offset =
          (simdgroup_col_thread_base + thread_col_in_simdgroup) *
          static_cast<uint>(gemv_tiling_thread_out_cols(gemv_tiling));

      uint output_row_start = thread_context.threadgroup_position.x *
              static_cast<uint>(params->output_rows_per_threadgroup) +
          output_block_row_offset;

      if (output_row_start >= params->out_vec_size) {
        return;
      }

      output_row_start = output_row_start +
                  static_cast<uint>(gemv_tiling_thread_out_rows(gemv_tiling)) <=
              params->out_vec_size
          ? output_row_start
          : params->out_vec_size -
              static_cast<uint>(gemv_tiling_thread_out_rows(gemv_tiling));

      const device T* thread_matrix =
          matrix + output_row_start * params->matrix_leading_dimension;

      const uniform<uint> input_block_stride =
          make_uniform(static_cast<uint>(input_columns_per_threadgroup));
      const uniform<uint> input_vector_length =
          make_uniform(params->in_vec_size);
      const uniform<uint> full_input_blocks =
          input_vector_length / input_block_stride;
      const uniform<uint> remaining_input_columns =
          input_vector_length - input_block_stride * full_input_blocks;

      for (uint input_block_index = 0; input_block_index < full_input_blocks;
           ++input_block_index) {
        {
          const device T* input_vector_row =
              a + batch_row * params->in_vec_size;
          METAL_PRAGMA_UNROLL
          for (uint input_col_offset = 0;
               input_col_offset < gemv_tiling_thread_out_cols(gemv_tiling);
               input_col_offset++) {
            vector_coefficients[input_col_offset] = static_cast<float>(
                input_vector_row[input_block_col_offset + input_col_offset]
            );
          }
        }

        uint matrix_row_offset = 0;
        METAL_PRAGMA_UNROLL
        for (uint output_row_offset = 0;
             output_row_offset < gemv_tiling_thread_out_rows(gemv_tiling);
             output_row_offset++) {
          METAL_PRAGMA_UNROLL
          for (uint input_col_offset = 0;
               input_col_offset < gemv_tiling_thread_out_cols(gemv_tiling);
               input_col_offset++) {
            matrix_values[input_col_offset] = thread_matrix
                [matrix_row_offset + input_block_col_offset +
                 input_col_offset];
          }

          METAL_PRAGMA_UNROLL
          for (uint input_col_offset = 0;
               input_col_offset < gemv_tiling_thread_out_cols(gemv_tiling);
               input_col_offset++) {
            accumulated_values[output_row_offset] +=
                static_cast<float>(matrix_values[input_col_offset]) *
                vector_coefficients[input_col_offset];
          }

          matrix_row_offset += params->matrix_leading_dimension;
        }

        input_block_col_offset += input_columns_per_threadgroup;
      }

      if (remaining_input_columns > 0) {
        {
          const device T* input_vector_row =
              a + batch_row * params->in_vec_size;
          if (input_block_col_offset +
                  static_cast<uint>(gemv_tiling_thread_out_cols(gemv_tiling)) <=
              input_vector_length) {
            METAL_PRAGMA_UNROLL
            for (uint input_col_offset = 0;
                 input_col_offset < gemv_tiling_thread_out_cols(gemv_tiling);
                 input_col_offset++) {
              vector_coefficients[input_col_offset] = static_cast<float>(
                  input_vector_row[input_block_col_offset + input_col_offset]
              );
            }
          } else {
            METAL_PRAGMA_UNROLL
            for (uint input_col_offset = 0;
                 input_col_offset < gemv_tiling_thread_out_cols(gemv_tiling);
                 input_col_offset++) {
              vector_coefficients[input_col_offset] =
                  input_block_col_offset +
                          static_cast<uint>(input_col_offset) <
                          input_vector_length
                      ? static_cast<float>(
                            input_vector_row
                                [input_block_col_offset + input_col_offset]
                        )
                      : 0.0f;
            }
          }
        }

        METAL_PRAGMA_UNROLL
        for (uint output_row_offset = 0;
             output_row_offset < gemv_tiling_thread_out_rows(gemv_tiling);
             output_row_offset++) {
          if (input_block_col_offset +
                  static_cast<uint>(gemv_tiling_thread_out_cols(gemv_tiling)) <=
              input_vector_length) {
            METAL_PRAGMA_UNROLL
            for (uint input_col_offset = 0;
                 input_col_offset < gemv_tiling_thread_out_cols(gemv_tiling);
                 input_col_offset++) {
              matrix_values[input_col_offset] = thread_matrix
                  [output_row_offset * params->matrix_leading_dimension +
                   input_block_col_offset + input_col_offset];
            }
          } else {
            METAL_PRAGMA_UNROLL
            for (uint input_col_offset = 0;
                 input_col_offset < gemv_tiling_thread_out_cols(gemv_tiling);
                 input_col_offset++) {
              matrix_values[input_col_offset] =
                  input_block_col_offset +
                          static_cast<uint>(input_col_offset) <
                          input_vector_length
                      ? thread_matrix
                            [output_row_offset *
                                 params->matrix_leading_dimension +
                             input_block_col_offset + input_col_offset]
                      : static_cast<T>(0);
            }
          }

          METAL_PRAGMA_UNROLL
          for (uint input_col_offset = 0;
               input_col_offset < gemv_tiling_thread_out_cols(gemv_tiling);
               input_col_offset++) {
            accumulated_values[output_row_offset] +=
                static_cast<float>(matrix_values[input_col_offset]) *
                vector_coefficients[input_col_offset];
          }
        }
      }

      // Simdgroup reduction across the K-split lanes.
      METAL_PRAGMA_UNROLL
      for (uint output_row_offset = 0;
           output_row_offset < gemv_tiling_thread_out_rows(gemv_tiling);
           output_row_offset++) {
        METAL_PRAGMA_UNROLL
        for (ushort simd_shuffle_offset =
                 (gemv_tiling_sg_thread_cols(gemv_tiling) / 2);
             simd_shuffle_offset >= 1;
             simd_shuffle_offset /= 2) {
          accumulated_values[output_row_offset] += simd_shuffle_down(
              accumulated_values[output_row_offset],
              simd_shuffle_offset
          );
        }
      }

      // Threadgroup reduction (only when tg_simd_cols > 1).
      if (gemv_tiling_tg_simd_cols(gemv_tiling) > 1) {
        const uint computed_output_rows_per_tg =
            static_cast<uint>(gemv_tiling_tg_simd_rows(gemv_tiling)) *
            static_cast<uint>(gemv_tiling_sg_thread_rows(gemv_tiling)) *
            static_cast<uint>(gemv_tiling_thread_out_rows(gemv_tiling));

        if (thread_col_in_simdgroup == 0) {
          threadgroup float* threadgroup_partial_accumulations =
              partial_shared +
              simdgroup_column_index *
                  (computed_output_rows_per_tg +
                   static_cast<uint>(
                       gemv_tiling_thread_out_rows(gemv_tiling))) +
              output_block_row_offset;
          METAL_PRAGMA_UNROLL
          for (uint output_row_offset = 0;
               output_row_offset < gemv_tiling_thread_out_rows(gemv_tiling);
               output_row_offset++) {
            threadgroup_partial_accumulations[output_row_offset] =
                accumulated_values[output_row_offset];
          }

          threadgroup_barrier(mem_flags::mem_none);

          if (simdgroup_column_index == 0) {
            threadgroup float* base_partial =
                partial_shared + output_block_row_offset;
            for (uint reduction_simdgroup_col = 1;
                 reduction_simdgroup_col <
                 gemv_tiling_tg_simd_cols(gemv_tiling);
                 reduction_simdgroup_col++) {
              METAL_PRAGMA_UNROLL
              for (uint output_row_offset = 0;
                   output_row_offset <
                   gemv_tiling_thread_out_rows(gemv_tiling);
                   output_row_offset++) {
                accumulated_values[output_row_offset] += base_partial
                    [reduction_simdgroup_col *
                         (computed_output_rows_per_tg +
                          static_cast<uint>(
                              gemv_tiling_thread_out_rows(gemv_tiling))) +
                     output_row_offset];
              }
            }
          }
        }
      }

      // Write outputs (scale / accumulate / bias).
      if (simdgroup_col_thread_base == 0 && thread_col_in_simdgroup == 0) {
        device T* output_row_values = d + batch_row * params->out_vec_size;
        METAL_PRAGMA_UNROLL
        for (uint output_row_offset = 0;
             output_row_offset < gemv_tiling_thread_out_rows(gemv_tiling);
             output_row_offset++) {
          T accumulated_c =
              static_cast<T>(accumulated_values[output_row_offset]);
          if (is_scale) {
            accumulated_c = static_cast<T>(params->ab_scale) * accumulated_c;
          }
          if (is_accumulate) {
            accumulated_c +=
                output_row_values[output_row_start + output_row_offset];
          }
          if (is_bias) {
            accumulated_c += output_bias[output_row_start + output_row_offset];
          }
          output_row_values[output_row_start + output_row_offset] =
              accumulated_c;
        }
      }

      if (use_hadamard) {
        threadgroup_barrier(mem_flags::mem_device);
        const uint sg_count = gemv_tiling_tg_simd_rows(gemv_tiling) *
            gemv_tiling_tg_simd_cols(gemv_tiling);
        const uint sg_id = thread_context.simdgroup_index;
        const ushort lane = thread_context.simd_lane_id;
        const uint tg_block_start = thread_context.threadgroup_position.x *
            params->output_rows_per_threadgroup;
        const uint stripes_per_tg =
            params->output_rows_per_threadgroup / METAL_SIMD_SIZE;
        device T* output_row_values = d + batch_row * params->out_vec_size;
        for (uint stripe = sg_id; stripe < stripes_per_tg;
             stripe += sg_count) {
          const uint global_col =
              tg_block_start + stripe * METAL_SIMD_SIZE + lane;
          if (global_col < params->out_vec_size) {
            T value = output_row_values[global_col];
            output_row_values[global_col] =
                simdgroup_output_random_hadamard_transform(
                    lane,
                    value,
                    rht_factors[global_col]
                );
          }
        }
      }
    } else {
      (void)gemv_tiling;
      (void)partial_shared;

      // Hardcoded Wide-tile layout: 8 simdgroups × 4 results/simdgroup. Tile-
      // awareness for quant is deferred (host pins `GemvTiling::Wide`); these
      // constants match the Wide tile and compile to immediates.
      constexpr uint packs_per_thread = BITS == 2 ? 1 : 2;
      constexpr uint num_simdgroups = 8;
      constexpr uint results_per_simdgroup = 4;
      constexpr uint pack_factor = qmv_pack_factor<BITS>();
      constexpr uint bytes_per_pack = qmv_bytes_per_pack<BITS>();
      constexpr uint values_per_thread = pack_factor * packs_per_thread;
      constexpr uint block_size = values_per_thread * METAL_SIMD_SIZE;
      constexpr uint scale_step_per_thread = GROUP_SIZE / values_per_thread;
      const device uint8_t* ws = b_packed;
      typedef float U;
      thread U x_thread[values_per_thread];
      thread U result[results_per_simdgroup] = {0};

      const uint in_vec_size_w =
          params->in_vec_size * bytes_per_pack / pack_factor;
      const uint in_vec_size_g = params->in_vec_size / GROUP_SIZE;
      const uint out_row =
          thread_context.threadgroup_position.x *
              (num_simdgroups * results_per_simdgroup) +
          thread_context.simdgroup_index * results_per_simdgroup;
      ws += out_row * in_vec_size_w +
          thread_context.simd_lane_id * packs_per_thread * bytes_per_pack;
      scales += out_row * in_vec_size_g +
          thread_context.simd_lane_id / scale_step_per_thread;

      uint zp_stride = 0;
      const device uint8_t* zps = nullptr;
      bool high_nibble = false;

      if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant) {
        biases += out_row * in_vec_size_g +
            thread_context.simd_lane_id / scale_step_per_thread;
      } else {
        if (BITS == 4) {
          zp_stride = (in_vec_size_g + 1) / 2;
          zps = zero_points + out_row * zp_stride;
          uint g_offset =
              thread_context.simd_lane_id / scale_step_per_thread;
          zps += g_offset / 2;
          high_nibble = (g_offset & 1);
        } else {
          zp_stride = in_vec_size_g;
          zps = zero_points + out_row * zp_stride;
          zps += thread_context.simd_lane_id / scale_step_per_thread;
        }
      }

      a += thread_context.threadgroup_position.y * params->in_vec_size +
          thread_context.simd_lane_id * values_per_thread;
      d += thread_context.threadgroup_position.y * params->out_vec_size +
          out_row;

      uint k = 0;
      for (; k + block_size <= params->in_vec_size; k += block_size) {
        U sum = load_vector<T, U, values_per_thread, BITS>(a, x_thread);

        {
          auto wl0 = static_cast<const device uint8_t*>(ws);
          auto wl1 = static_cast<const device uint8_t*>(ws + in_vec_size_w);
          auto wl2 =
              static_cast<const device uint8_t*>(ws + 2 * in_vec_size_w);
          auto wl3 =
              static_cast<const device uint8_t*>(ws + 3 * in_vec_size_w);

          U s0 = static_cast<U>(scales[0]);
          U s1 = static_cast<U>(scales[in_vec_size_g]);
          U s2 = static_cast<U>(scales[2 * in_vec_size_g]);
          U s3 = static_cast<U>(scales[3 * in_vec_size_g]);

          if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant) {
            U b0 = static_cast<U>(biases[0]);
            U b1 = static_cast<U>(biases[in_vec_size_g]);
            U b2 = static_cast<U>(biases[2 * in_vec_size_g]);
            U b3 = static_cast<U>(biases[3 * in_vec_size_g]);
            result[0] +=
                qdot<U, values_per_thread, BITS>(wl0, x_thread, s0, b0, sum);
            result[1] +=
                qdot<U, values_per_thread, BITS>(wl1, x_thread, s1, b1, sum);
            result[2] +=
                qdot<U, values_per_thread, BITS>(wl2, x_thread, s2, b2, sum);
            result[3] +=
                qdot<U, values_per_thread, BITS>(wl3, x_thread, s3, b3, sum);
          } else {
            uchar4 zp_bytes = uchar4(
                zps[0],
                zps[zp_stride],
                zps[2 * zp_stride],
                zps[3 * zp_stride]
            );
            uchar4 zp_nibbles;
            if (BITS == 4) {
              const uint8_t shift = high_nibble ? 4u : 0u;
              zp_nibbles = (zp_bytes >> shift) & uchar4(0x0F);
            } else {
              zp_nibbles = zp_bytes;
            }
            result[0] += qdot<U, values_per_thread, BITS>(
                wl0,
                x_thread,
                s0,
                -s0 * static_cast<U>(zp_nibbles.x),
                sum
            );
            result[1] += qdot<U, values_per_thread, BITS>(
                wl1,
                x_thread,
                s1,
                -s1 * static_cast<U>(zp_nibbles.y),
                sum
            );
            result[2] += qdot<U, values_per_thread, BITS>(
                wl2,
                x_thread,
                s2,
                -s2 * static_cast<U>(zp_nibbles.z),
                sum
            );
            result[3] += qdot<U, values_per_thread, BITS>(
                wl3,
                x_thread,
                s3,
                -s3 * static_cast<U>(zp_nibbles.w),
                sum
            );
          }
        }

        ws += block_size * bytes_per_pack / pack_factor;
        scales += block_size / GROUP_SIZE;
        if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant) {
          biases += block_size / GROUP_SIZE;
        } else {
          if (BITS == 4) {
            zps += (block_size / GROUP_SIZE) / 2;
          } else {
            zps += block_size / GROUP_SIZE;
          }
        }
        a += block_size;
      }

      const uint thread_offset =
          thread_context.simd_lane_id * values_per_thread;
      const int remaining = (k + thread_offset < params->in_vec_size)
          ? min(
                static_cast<int>(params->in_vec_size - k - thread_offset),
                static_cast<int>(values_per_thread))
          : 0;
      if (remaining > 0) {
        U sum = load_vector_safe<T, U, values_per_thread, BITS>(
            a,
            x_thread,
            remaining
        );

        auto wl0 = static_cast<const device uint8_t*>(ws);
        auto wl1 = static_cast<const device uint8_t*>(ws + in_vec_size_w);
        auto wl2 = static_cast<const device uint8_t*>(ws + 2 * in_vec_size_w);
        auto wl3 = static_cast<const device uint8_t*>(ws + 3 * in_vec_size_w);

        U s0 = static_cast<U>(scales[0]);
        U s1 = static_cast<U>(scales[in_vec_size_g]);
        U s2 = static_cast<U>(scales[2 * in_vec_size_g]);
        U s3 = static_cast<U>(scales[3 * in_vec_size_g]);

        if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant) {
          U b0 = static_cast<U>(biases[0]);
          U b1 = static_cast<U>(biases[in_vec_size_g]);
          U b2 = static_cast<U>(biases[2 * in_vec_size_g]);
          U b3 = static_cast<U>(biases[3 * in_vec_size_g]);
          result[0] += qdot_safe<U, values_per_thread, BITS>(
              wl0, x_thread, s0, b0, sum, remaining);
          result[1] += qdot_safe<U, values_per_thread, BITS>(
              wl1, x_thread, s1, b1, sum, remaining);
          result[2] += qdot_safe<U, values_per_thread, BITS>(
              wl2, x_thread, s2, b2, sum, remaining);
          result[3] += qdot_safe<U, values_per_thread, BITS>(
              wl3, x_thread, s3, b3, sum, remaining);
        } else {
          uchar4 zp_bytes = uchar4(
              zps[0], zps[zp_stride], zps[2 * zp_stride], zps[3 * zp_stride]);
          uchar4 zp_nibbles;
          if (BITS == 4) {
            const uint8_t shift = high_nibble ? 4u : 0u;
            zp_nibbles = (zp_bytes >> shift) & uchar4(0x0F);
          } else {
            zp_nibbles = zp_bytes;
          }
          result[0] += qdot_safe<U, values_per_thread, BITS>(
              wl0,
              x_thread,
              s0,
              -s0 * static_cast<U>(zp_nibbles.x),
              sum,
              remaining);
          result[1] += qdot_safe<U, values_per_thread, BITS>(
              wl1,
              x_thread,
              s1,
              -s1 * static_cast<U>(zp_nibbles.y),
              sum,
              remaining);
          result[2] += qdot_safe<U, values_per_thread, BITS>(
              wl2,
              x_thread,
              s2,
              -s2 * static_cast<U>(zp_nibbles.z),
              sum,
              remaining);
          result[3] += qdot_safe<U, values_per_thread, BITS>(
              wl3,
              x_thread,
              s3,
              -s3 * static_cast<U>(zp_nibbles.w),
              sum,
              remaining);
        }
      }

      for (uint row = 0; row < results_per_simdgroup; row++) {
        result[row] = simd_sum(result[row]);
      }

      // Fused output epilogue: optional scale, then optional accumulate / bias.
      if (thread_context.simd_lane_id == 0) {
        for (uint row = 0; row < results_per_simdgroup; row++) {
          U value = result[row];
          if (is_scale) {
            value = static_cast<U>(params->ab_scale) * value;
          }
          const uint global_row = out_row + row;
          if (is_accumulate && global_row < params->out_vec_size) {
            value += static_cast<U>(d[row]);
          }
          if (is_bias && global_row < params->out_vec_size) {
            value += static_cast<U>(output_bias[global_row]);
          }
          result[row] = value;
        }
      }

      if (use_hadamard) {
        if (thread_context.simd_lane_id == 0) {
          for (uint row = 0; row < results_per_simdgroup; row++) {
            result_shared
                [thread_context.simdgroup_index * results_per_simdgroup +
                 row] = result[row];
          }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (thread_context.simdgroup_index == 0) {
          uint global_out_idx = thread_context.threadgroup_position.x * 32 +
              thread_context.simd_lane_id;
          if (global_out_idx < params->out_vec_size) {
            d[thread_context.simd_lane_id] =
                simdgroup_output_random_hadamard_transform(
                    static_cast<ushort>(thread_context.simd_lane_id),
                    static_cast<T>(
                        result_shared[thread_context.simd_lane_id]),
                    rht_factors[global_out_idx]
                );
          }
        }
      } else {
        if (thread_context.simd_lane_id == 0) {
          for (uint row = 0; row < results_per_simdgroup; row++) {
            d[row] = static_cast<T>(result[row]);
          }
        }
      }
    }
  }
};

} // namespace gemv
} // namespace uzu
