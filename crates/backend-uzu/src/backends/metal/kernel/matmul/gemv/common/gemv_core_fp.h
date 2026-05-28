#pragma once

// Out-of-class definition for GemvCore::run_fp. Include via gemv_core.h.

namespace uzu {
namespace gemv {

template <typename T, GemmBPrologueKind B_PROLOGUE, int BITS, int GROUP_SIZE>
METAL_FUNC void GemvCore<T, B_PROLOGUE, BITS, GROUP_SIZE>::run_fp(
    const device T* matrix,
    const device T* input,
    device T* output,
    const device int32_t* hadamard_factors,
    const device T* output_bias,
    const constant uzu::matmul::GemvParams* params,
    GemmDTransform output_transform,
    GemvTile tile,
    uint threadgroup_index_x,
    uint threadgroup_index_y,
    threadgroup float* threadgroup_memory,
    const thread ThreadContext& thread_context
) {
  const uint in_vec_size = params->in_vec_size;
  const uint out_vec_size = params->out_vec_size;
  const uint batch_size = params->batch_size;
  const uint matrix_leading_dimension = params->matrix_leading_dimension;
  const uint output_rows_per_threadgroup = params->output_rows_per_threadgroup;
  const float output_scale = params->ab_scale;
  const bool is_scale = output_transform.contains(GemmDTransform::SCALE);
  const bool is_accumulate =
      output_transform.contains(GemmDTransform::ACCUMULATE);
  const bool is_bias = output_transform.contains(GemmDTransform::BIAS);
  const bool use_hadamard = output_transform.contains(GemmDTransform::RHT);
  const uint tg_simd_rows = tile.tg_simd_rows;
  const uint tg_simd_cols = tile.tg_simd_cols;
  const uint sg_thread_rows = tile.sg_thread_rows;
  const uint sg_thread_cols = tile.sg_thread_cols;
  const uint thread_out_rows = tile.thread_out_rows;
  const uint thread_out_cols = tile.thread_out_cols;

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

  output_row_start = output_row_start + uint(thread_out_rows) <= out_vec_size
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
          static_cast<T>(accumulated_values[output_row_offset]);
      if (is_scale) {
        accumulated_c = static_cast<T>(output_scale) * accumulated_c;
      }
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
}

} // namespace gemv
} // namespace uzu
