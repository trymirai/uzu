#include <metal_stdlib>
#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../../common/thread_context.h"
#include "../../hadamard_transform/hadamard_transform.h"
#include "../../generated/quantization_method.h"
#include "../../generated/gemm.h"
#include "../../generated/matmul.h"
#include "../common/qdot.h"
#include "../common/quant_pack.h"

#include <metal_simdgroup>

using namespace metal;
using namespace uzu::quantization_method;
using namespace uzu::gemm;

// Upper bound for the full-precision threadgroup reduction scratch (in floats).
// Max config: tg_simd_cols=16, output_rows_per_tg=16, thread_out_rows=4
// → 16 * (16 + 4) = 320. Unused when tg_simd_cols == 1.
#define GEMV_MAX_THREADGROUP_MEMORY 320

// 0-safe wrappers: BITS == 0 denotes the full-precision (quant-off) path, so we
// must not instantiate the quant pack helpers with a zero bit-width.
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

// Unified GEMV: a single simdgroup-reduction matrix-vector kernel that handles
// both full-precision (BITS == 0) and group-quantized weights, selected at
// compile time. The two paths keep their own proven inner loops; only the entry
// point, signature, and dispatch are shared.
template <typename T, uint GROUP_SIZE, uint BITS>
VARIANTS(T, float, half, bfloat)
VARIANTS(GROUP_SIZE, 0, 32, 64, 128)
VARIANTS(BITS, 0, 4, 8)
CONSTRAINT((BITS == 0) == (GROUP_SIZE == 0))
KERNEL(Gemv)(
    const device uint32_t* weights,
    const device T* scales OPTIONAL(BITS != 0),
    const device uint8_t* zero_points
        OPTIONAL(BITS != 0 && quant_method == QuantizationMethod::ScaleZeroPoint),
    const device T* biases
        OPTIONAL(BITS != 0 && quant_method == QuantizationMethod::ScaleBias),
    const device T* input,
    device T* output,
    const device int32_t* hadamard_factors
        OPTIONAL(output_transform.contains(GemmDTransform::RHT)),
    const device T* output_bias
        OPTIONAL(output_transform.contains(GemmDTransform::BIAS)),
    const constant uzu::matmul::GemvParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const QuantizationMethod quant_method SPECIALIZE,
    const GemmDTransform output_transform SPECIALIZE,
    const uint tg_simd_rows SPECIALIZE,
    const uint tg_simd_cols SPECIALIZE,
    const uint sg_thread_rows SPECIALIZE,
    const uint sg_thread_cols SPECIALIZE,
    const uint thread_out_rows SPECIALIZE,
    const uint thread_out_cols SPECIALIZE,
    threadgroup float threadgroup_memory[GEMV_MAX_THREADGROUP_MEMORY],
    threadgroup float shared_results[METAL_SIMD_SIZE],
    const uint group_index_x GROUPS(group_count_x),
    const uint group_index_y GROUPS(group_count_y),
    const uint thread_index_x THREADS(METAL_SIMD_SIZE),
    const uint thread_index_y THREADS(8),
    const uint thread_index_z THREADS(1),
    const ThreadContext thread_context
) {
  (void)thread_index_z;

  const uint in_vec_size = params->in_vec_size;
  const uint out_vec_size = params->out_vec_size;
  const uint batch_size = params->batch_size;
  const uint matrix_leading_dimension = params->matrix_leading_dimension;
  const uint output_rows_per_threadgroup = params->output_rows_per_threadgroup;
  const float output_scale = params->ab_scale;
  const bool is_accumulate = output_transform.contains(GemmDTransform::ACCUMULATE);
  const bool is_bias = output_transform.contains(GemmDTransform::BIAS);
  const bool use_hadamard = output_transform.contains(GemmDTransform::RHT);

  // Compile-time prologue, mirroring gemm's SimdgroupMmaCore dispatch. Full
  // precision is encoded as BITS == 0 (a quantized prologue always carries a
  // nonzero bit width); unlike gemm the prologue can't be a template enum here
  // because the CPU reference kernel can't take an enum const generic on stable
  // Rust, so it is derived from BITS. The quantized sub-kind (scale+bias vs
  // scale+zero-point) is selected at runtime via the `quant_method` constant.
  constexpr GemmBPrologueKind B_PROLOGUE = (BITS == 0)
      ? GemmBPrologueKind::FullPrecision
      : GemmBPrologueKind::ScaleBiasDequant;

  if constexpr (B_PROLOGUE == GemmBPrologueKind::FullPrecision) {
    // =========================== Full precision ===========================
    // Dense weights laid out as [out_vec_size, in_vec_size]; tunable tile
    // (tg/sg/thread counts come from the host heuristic via SPECIALIZE).
    const device T* matrix = reinterpret_cast<const device T*>(weights);
    const uint threadgroup_index_x = group_index_x;
    const uint threadgroup_index_y = group_index_y;

    if (output_rows_per_threadgroup <= 0) {
      return;
    }

    // Kill excess simdgroups beyond what this tile uses.
    if (thread_context.simdgroup_index >= tg_simd_rows * tg_simd_cols) {
      return;
    }

    const uint threads_per_threadgroup_col = tg_simd_cols * sg_thread_cols;
    const uint input_columns_per_threadgroup =
        threads_per_threadgroup_col * thread_out_cols;

    const uint batch_row = static_cast<uint>(threadgroup_index_y);
    if (batch_row >= batch_size) {
      return;
    }

    thread float accumulated_values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    thread T matrix_values[4];
    thread float vector_coefficients[4];

    const uint thread_row_in_simdgroup =
        sg_thread_cols != 32
            ? uint(thread_context.simd_lane_id) / uint(sg_thread_cols)
            : 0;
    const int thread_col_in_simdgroup =
        sg_thread_cols != 32
            ? uint(thread_context.simd_lane_id) % uint(sg_thread_cols)
            : uint(thread_context.simd_lane_id);

    const int simdgroup_column_index =
        tg_simd_cols != 1 ? uint(thread_context.simdgroup_index % tg_simd_cols)
                          : 0;

    const uint simdgroup_row_thread_base =
        tg_simd_cols != 1
            ? uint(sg_thread_rows) *
                  uint(thread_context.simdgroup_index / tg_simd_cols)
            : uint(sg_thread_rows) * uint(thread_context.simdgroup_index);
    const uint simdgroup_col_thread_base =
        tg_simd_cols != 1
            ? uint(sg_thread_cols) *
                  uint(thread_context.simdgroup_index % tg_simd_cols)
            : 0;

    uint output_block_row_offset =
        (simdgroup_row_thread_base + thread_row_in_simdgroup) *
        uint(thread_out_rows);
    uint input_block_col_offset =
        (simdgroup_col_thread_base + thread_col_in_simdgroup) *
        uint(thread_out_cols);

    uint output_row_start =
        uint(threadgroup_index_x) * uint(output_rows_per_threadgroup) +
        output_block_row_offset;

    if (output_row_start >= out_vec_size) {
      return;
    }

    output_row_start =
        output_row_start + uint(thread_out_rows) <= out_vec_size
            ? output_row_start
            : out_vec_size - uint(thread_out_rows);

    const device T* thread_matrix =
        matrix + output_row_start * matrix_leading_dimension;

    const uniform<uint> input_block_stride =
        make_uniform(uint(input_columns_per_threadgroup));
    const uniform<uint> input_vector_length = make_uniform(in_vec_size);
    const uniform<uint> full_input_blocks =
        input_vector_length / input_block_stride;
    const uniform<uint> remaining_input_columns =
        input_vector_length - input_block_stride * full_input_blocks;

    for (uint input_block_index = 0; input_block_index < full_input_blocks;
         ++input_block_index) {
      {
        const device T* input_vector_row = input + batch_row * in_vec_size;
        METAL_PRAGMA_UNROLL
        for (uint input_col_offset = 0; input_col_offset < thread_out_cols;
             input_col_offset++) {
          vector_coefficients[input_col_offset] = static_cast<float>(
              input_vector_row[input_block_col_offset + input_col_offset]
          );
        }
      }

      uint matrix_row_offset = 0;
      METAL_PRAGMA_UNROLL
      for (uint output_row_offset = 0; output_row_offset < thread_out_rows;
           output_row_offset++) {
        METAL_PRAGMA_UNROLL
        for (uint input_col_offset = 0; input_col_offset < thread_out_cols;
             input_col_offset++) {
          matrix_values[input_col_offset] = thread_matrix
              [matrix_row_offset + input_block_col_offset + input_col_offset];
        }

        METAL_PRAGMA_UNROLL
        for (uint input_col_offset = 0; input_col_offset < thread_out_cols;
             input_col_offset++) {
          accumulated_values[output_row_offset] +=
              static_cast<float>(matrix_values[input_col_offset]) *
              vector_coefficients[input_col_offset];
        }

        matrix_row_offset += matrix_leading_dimension;
      }

      input_block_col_offset += input_columns_per_threadgroup;
    }

    if (remaining_input_columns > 0) {
      {
        const device T* input_vector_row = input + batch_row * in_vec_size;
        if (input_block_col_offset + uint(thread_out_cols) <=
            input_vector_length) {
          METAL_PRAGMA_UNROLL
          for (uint input_col_offset = 0; input_col_offset < thread_out_cols;
               input_col_offset++) {
            vector_coefficients[input_col_offset] = static_cast<float>(
                input_vector_row[input_block_col_offset + input_col_offset]
            );
          }
        } else {
          METAL_PRAGMA_UNROLL
          for (uint input_col_offset = 0; input_col_offset < thread_out_cols;
               input_col_offset++) {
            vector_coefficients[input_col_offset] =
                input_block_col_offset + uint(input_col_offset) <
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
      for (uint output_row_offset = 0; output_row_offset < thread_out_rows;
           output_row_offset++) {
        if (input_block_col_offset + uint(thread_out_cols) <=
            input_vector_length) {
          METAL_PRAGMA_UNROLL
          for (uint input_col_offset = 0; input_col_offset < thread_out_cols;
               input_col_offset++) {
            matrix_values[input_col_offset] = thread_matrix
                [output_row_offset * matrix_leading_dimension +
                 input_block_col_offset + input_col_offset];
          }
        } else {
          METAL_PRAGMA_UNROLL
          for (uint input_col_offset = 0; input_col_offset < thread_out_cols;
               input_col_offset++) {
            matrix_values[input_col_offset] =
                input_block_col_offset + uint(input_col_offset) <
                        input_vector_length
                    ? thread_matrix
                          [output_row_offset * matrix_leading_dimension +
                           input_block_col_offset + input_col_offset]
                    : T(0);
          }
        }

        METAL_PRAGMA_UNROLL
        for (uint input_col_offset = 0; input_col_offset < thread_out_cols;
             input_col_offset++) {
          accumulated_values[output_row_offset] +=
              static_cast<float>(matrix_values[input_col_offset]) *
              vector_coefficients[input_col_offset];
        }
      }
    }

    // Simdgroup reduction across the K-split lanes.
    METAL_PRAGMA_UNROLL
    for (uint output_row_offset = 0; output_row_offset < thread_out_rows;
         output_row_offset++) {
      METAL_PRAGMA_UNROLL
      for (ushort simd_shuffle_offset = (sg_thread_cols / 2);
           simd_shuffle_offset >= 1;
           simd_shuffle_offset /= 2) {
        accumulated_values[output_row_offset] += simd_shuffle_down(
            accumulated_values[output_row_offset],
            simd_shuffle_offset
        );
      }
    }

    // Threadgroup reduction (only when tg_simd_cols > 1).
    if (tg_simd_cols > 1) {
      const uint computed_output_rows_per_tg =
          uint(tg_simd_rows) * uint(sg_thread_rows) * uint(thread_out_rows);

      if (thread_col_in_simdgroup == 0) {
        threadgroup float* threadgroup_partial_accumulations =
            threadgroup_memory +
            simdgroup_column_index *
                (computed_output_rows_per_tg + uint(thread_out_rows)) +
            output_block_row_offset;
        METAL_PRAGMA_UNROLL
        for (uint output_row_offset = 0; output_row_offset < thread_out_rows;
             output_row_offset++) {
          threadgroup_partial_accumulations[output_row_offset] =
              accumulated_values[output_row_offset];
        }

        threadgroup_barrier(mem_flags::mem_none);

        if (simdgroup_column_index == 0) {
          threadgroup float* base_partial =
              threadgroup_memory + output_block_row_offset;
          for (uint reduction_simdgroup_col = 1;
               reduction_simdgroup_col < tg_simd_cols;
               reduction_simdgroup_col++) {
            METAL_PRAGMA_UNROLL
            for (uint output_row_offset = 0;
                 output_row_offset < thread_out_rows;
                 output_row_offset++) {
              accumulated_values[output_row_offset] += base_partial
                  [reduction_simdgroup_col *
                       (computed_output_rows_per_tg + uint(thread_out_rows)) +
                   output_row_offset];
            }
          }
        }
      }
    }

    // Write outputs (scale / accumulate / bias).
    if (simdgroup_col_thread_base == 0 && thread_col_in_simdgroup == 0) {
      device T* output_row_values = output + batch_row * out_vec_size;
      METAL_PRAGMA_UNROLL
      for (uint output_row_offset = 0; output_row_offset < thread_out_rows;
           output_row_offset++) {
        T accumulated_c =
            static_cast<T>(output_scale) *
            static_cast<T>(accumulated_values[output_row_offset]);
        if (is_accumulate) {
          accumulated_c +=
              output_row_values[output_row_start + output_row_offset];
        }
        if (is_bias) {
          accumulated_c += output_bias[output_row_start + output_row_offset];
        }
        output_row_values[output_row_start + output_row_offset] = accumulated_c;
      }
    }

    if (use_hadamard) {
      threadgroup_barrier(mem_flags::mem_device);
      const uint sg_count = tg_simd_rows * tg_simd_cols;
      const uint sg_id = thread_context.simdgroup_index;
      const ushort lane = thread_context.simd_lane_id;
      const uint tg_block_start =
          threadgroup_index_x * output_rows_per_threadgroup;
      const uint stripes_per_tg = output_rows_per_threadgroup / METAL_SIMD_SIZE;
      device T* output_row_values = output + batch_row * out_vec_size;
      for (uint stripe = sg_id; stripe < stripes_per_tg; stripe += sg_count) {
        const uint global_col =
            tg_block_start + stripe * METAL_SIMD_SIZE + lane;
        if (global_col < out_vec_size) {
          T value = output_row_values[global_col];
          output_row_values[global_col] =
              simdgroup_output_random_hadamard_transform(
                  lane,
                  value,
                  hadamard_factors[global_col]
              );
        }
      }
    }
  } else {

  // ============================== Quantized ==============================
  const uint simd_lane = thread_index_x;
  const uint simd_group = thread_index_y;
  const uint batch_idx = group_index_x;
  const uint out_block_idx = group_index_y;

  constexpr uint packs_per_thread = BITS == 2 ? 1 : 2;
  constexpr uint num_simdgroups = 8;
  constexpr uint results_per_simdgroup = 4;
  constexpr uint pack_factor = qmv_pack_factor<BITS>();
  constexpr uint bytes_per_pack = qmv_bytes_per_pack<BITS>();
  constexpr uint values_per_thread = pack_factor * packs_per_thread;
  constexpr uint block_size = values_per_thread * METAL_SIMD_SIZE;
  constexpr uint scale_step_per_thread = GROUP_SIZE / values_per_thread;
  const device uint8_t* ws = (const device uint8_t*)weights;
  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const uint in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const uint in_vec_size_g = in_vec_size / GROUP_SIZE;
  const uint out_row =
      out_block_idx * (num_simdgroups * results_per_simdgroup) +
      simd_group * results_per_simdgroup;
  ws += out_row * in_vec_size_w + simd_lane * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lane / scale_step_per_thread;

  uint zp_stride = 0;
  const device uint8_t* zps = nullptr;
  bool high_nibble = false;

  if (quant_method == QuantizationMethod::ScaleBias) {
    biases += out_row * in_vec_size_g + simd_lane / scale_step_per_thread;
  } else {
    if (BITS == 4) {
      zp_stride = (in_vec_size_g + 1) / 2;
      zps = zero_points + out_row * zp_stride;
      uint g_offset = simd_lane / scale_step_per_thread;
      zps += g_offset / 2;
      high_nibble = (g_offset & 1);
    } else {
      zp_stride = in_vec_size_g;
      zps = zero_points + out_row * zp_stride;
      zps += simd_lane / scale_step_per_thread;
    }
  }

  input += batch_idx * in_vec_size + simd_lane * values_per_thread;
  output += batch_idx * out_vec_size + out_row;

  uint k = 0;
  for (; k + block_size <= in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, BITS>(input, x_thread);

    {
      auto wl0 = (const device uint8_t*)(ws);
      auto wl1 = (const device uint8_t*)(ws + in_vec_size_w);
      auto wl2 = (const device uint8_t*)(ws + 2 * in_vec_size_w);
      auto wl3 = (const device uint8_t*)(ws + 3 * in_vec_size_w);

      U s0 = static_cast<U>(scales[0]);
      U s1 = static_cast<U>(scales[in_vec_size_g]);
      U s2 = static_cast<U>(scales[2 * in_vec_size_g]);
      U s3 = static_cast<U>(scales[3 * in_vec_size_g]);

      if (quant_method == QuantizationMethod::ScaleBias) {
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
    if (quant_method == QuantizationMethod::ScaleBias) {
      biases += block_size / GROUP_SIZE;
    } else {
      if (BITS == 4) {
        zps += (block_size / GROUP_SIZE) / 2;
      } else {
        zps += block_size / GROUP_SIZE;
      }
    }
    input += block_size;
  }

  const uint thread_offset = simd_lane * values_per_thread;
  const int remaining =
      (k + thread_offset < in_vec_size)
          ? min(int(in_vec_size - k - thread_offset), int(values_per_thread))
          : 0;
  if (remaining > 0) {
    U sum = load_vector_safe<T, U, values_per_thread, BITS>(
        input,
        x_thread,
        remaining
    );

    auto wl0 = (const device uint8_t*)(ws);
    auto wl1 = (const device uint8_t*)(ws + in_vec_size_w);
    auto wl2 = (const device uint8_t*)(ws + 2 * in_vec_size_w);
    auto wl3 = (const device uint8_t*)(ws + 3 * in_vec_size_w);

    U s0 = static_cast<U>(scales[0]);
    U s1 = static_cast<U>(scales[in_vec_size_g]);
    U s2 = static_cast<U>(scales[2 * in_vec_size_g]);
    U s3 = static_cast<U>(scales[3 * in_vec_size_g]);

    if (quant_method == QuantizationMethod::ScaleBias) {
      U b0 = static_cast<U>(biases[0]);
      U b1 = static_cast<U>(biases[in_vec_size_g]);
      U b2 = static_cast<U>(biases[2 * in_vec_size_g]);
      U b3 = static_cast<U>(biases[3 * in_vec_size_g]);
      result[0] += qdot_safe<U, values_per_thread, BITS>(
          wl0,
          x_thread,
          s0,
          b0,
          sum,
          remaining
      );
      result[1] += qdot_safe<U, values_per_thread, BITS>(
          wl1,
          x_thread,
          s1,
          b1,
          sum,
          remaining
      );
      result[2] += qdot_safe<U, values_per_thread, BITS>(
          wl2,
          x_thread,
          s2,
          b2,
          sum,
          remaining
      );
      result[3] += qdot_safe<U, values_per_thread, BITS>(
          wl3,
          x_thread,
          s3,
          b3,
          sum,
          remaining
      );
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
      result[0] += qdot_safe<U, values_per_thread, BITS>(
          wl0,
          x_thread,
          s0,
          -s0 * static_cast<U>(zp_nibbles.x),
          sum,
          remaining
      );
      result[1] += qdot_safe<U, values_per_thread, BITS>(
          wl1,
          x_thread,
          s1,
          -s1 * static_cast<U>(zp_nibbles.y),
          sum,
          remaining
      );
      result[2] += qdot_safe<U, values_per_thread, BITS>(
          wl2,
          x_thread,
          s2,
          -s2 * static_cast<U>(zp_nibbles.z),
          sum,
          remaining
      );
      result[3] += qdot_safe<U, values_per_thread, BITS>(
          wl3,
          x_thread,
          s3,
          -s3 * static_cast<U>(zp_nibbles.w),
          sum,
          remaining
      );
    }
  }

  for (uint row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
  }

  // Fused output epilogue: scale, then optional accumulate / bias.
  if (simd_lane == 0) {
    for (uint row = 0; row < results_per_simdgroup; row++) {
      U value = static_cast<U>(output_scale) * result[row];
      const uint global_row = out_row + row;
      if (is_accumulate && global_row < out_vec_size) {
        value += static_cast<U>(output[row]);
      }
      if (is_bias && global_row < out_vec_size) {
        value += static_cast<U>(output_bias[global_row]);
      }
      result[row] = value;
    }
  }

  if (use_hadamard) {
    if (simd_lane == 0) {
      for (uint row = 0; row < results_per_simdgroup; row++) {
        shared_results[simd_group * results_per_simdgroup + row] = result[row];
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
      uint global_out_idx = out_block_idx * 32 + simd_lane;
      if (global_out_idx < out_vec_size) {
        output[simd_lane] = simdgroup_output_random_hadamard_transform(
            static_cast<ushort>(simd_lane),
            T(shared_results[simd_lane]),
            hadamard_factors[global_out_idx]
        );
      }
    }
  } else {
    if (simd_lane == 0) {
      for (uint row = 0; row < results_per_simdgroup; row++) {
        output[row] = static_cast<T>(result[row]);
      }
    }
  }
  }
}
