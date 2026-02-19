#include "../../../common/utils.h"
#include "../../../definitions.metal"

#include <metal_simdgroup>

using namespace metal;

// Upper bound for threadgroup memory (in floats).
// Max config: tg_simd_cols=16, output_rows_per_tg=16, thread_out_rows=4
// → 16 * (16 + 4) = 320. Unused when tg_simd_cols == 1.
#define GEMV_MAX_TG_MEMORY 320

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MatmulGemv)(
    const device T* matrix,
    const device T* input_vector,
    const device T* output_source OPTIONAL(apply_output_scale_and_accumulate),
    device T* output_vector,
    const constant int& input_dimension,
    const constant int& output_dimension,
    const constant int& matrix_leading_dimension,
    const constant float& output_scale,
    const constant float& output_accumulate_scale,
    const constant int* batch_shape,
    const constant int* vector_batch_stride,
    const constant int* matrix_batch_stride,
    const constant int* output_source_batch_stride,
    const constant int& output_source_stride,
    const constant int& batch_rows,
    const constant int& output_rows_per_threadgroup,
    threadgroup float threadgroup_memory[GEMV_MAX_TG_MEMORY],
    const uint tg_simd_rows SPECIALIZE,
    const uint tg_simd_cols SPECIALIZE,
    const uint sg_thread_rows SPECIALIZE,
    const uint sg_thread_cols SPECIALIZE,
    const uint thread_out_rows SPECIALIZE,
    const uint thread_out_cols SPECIALIZE,
    const bool apply_output_scale_and_accumulate SPECIALIZE,
    const uint threadgroup_index_x GROUPS((output_dimension + output_rows_per_threadgroup - 1) / output_rows_per_threadgroup),
    const uint threadgroup_index_y GROUPS(batch_rows),
    const uint threadgroup_index_z GROUPS(batch_shape[0]),
    const uint thread_index_x THREADS(32),
    const uint thread_index_y THREADS(16),
    const uint thread_index_z THREADS(1),
    const Simd simd
) {
  if (output_rows_per_threadgroup <= 0) {
    return;
  }

  // Simdgroup guard — kill excess simdgroups beyond what this tile uses
  if (simd.group_idx >= tg_simd_rows * tg_simd_cols) {
    return;
  }

  // Derived constants from SPECIALIZE params
  const int threads_per_threadgroup_col = tg_simd_cols * sg_thread_cols;
  const int input_columns_per_threadgroup =
      threads_per_threadgroup_col * thread_out_cols;

  // Batch setup (from run_matmul_gemv_shape)
  const uint3 threadgroup_position =
      uint3(threadgroup_index_x, threadgroup_index_y, threadgroup_index_z);

  const int batch_row = static_cast<int>(threadgroup_position.y);
  if (batch_row >= batch_rows) {
    return;
  }

  // Advance pointers by batch
  const device T* batch_input_vector =
      input_vector + threadgroup_position.z * vector_batch_stride[0];
  const device T* batch_matrix =
      matrix + threadgroup_position.z * matrix_batch_stride[0];
  const device T* batch_output_source = output_source;
  if (apply_output_scale_and_accumulate) {
    batch_output_source +=
        threadgroup_position.z * output_source_batch_stride[0];
  }
  device T* batch_output_vector =
      output_vector + threadgroup_position.z * batch_rows * output_dimension;

  // Thread local accumulation results
  thread float accumulated_values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  thread T matrix_values[4];
  thread float vector_coefficients[4];

  const int thread_row_in_simdgroup =
      sg_thread_cols != 32 ? int(simd.lane_idx) / int(sg_thread_cols) : 0;
  const int thread_col_in_simdgroup =
      sg_thread_cols != 32 ? int(simd.lane_idx) % int(sg_thread_cols)
                           : int(simd.lane_idx);

  const int simdgroup_column_index =
      tg_simd_cols != 1 ? int(simd.group_idx % tg_simd_cols) : 0;

  const int simdgroup_row_thread_base =
      tg_simd_cols != 1
          ? int(sg_thread_rows) * int(simd.group_idx / tg_simd_cols)
          : int(sg_thread_rows) * int(simd.group_idx);
  const int simdgroup_col_thread_base =
      tg_simd_cols != 1
          ? int(sg_thread_cols) * int(simd.group_idx % tg_simd_cols)
          : 0;

  int output_block_row_offset =
      (simdgroup_row_thread_base + thread_row_in_simdgroup) *
      int(thread_out_rows);
  int input_block_col_offset =
      (simdgroup_col_thread_base + thread_col_in_simdgroup) *
      int(thread_out_cols);

  // Block position
  int output_row_start =
      int(threadgroup_position.x) * int(output_rows_per_threadgroup) +
      output_block_row_offset;

  // Exit simdgroup if rows out of bound
  if (output_row_start >= output_dimension)
    return;

  // Adjust tail simdgroup to ensure in bound reads
  output_row_start = output_row_start + int(thread_out_rows) <= output_dimension
                         ? output_row_start
                         : output_dimension - int(thread_out_rows);

  // Advance matrix
  const device T* thread_matrix =
      batch_matrix + output_row_start * matrix_leading_dimension;

  const uniform<int> input_block_stride =
      make_uniform(int(input_columns_per_threadgroup));
  const uniform<int> input_vector_length = make_uniform(input_dimension);
  const uniform<int> full_input_blocks =
      input_vector_length / input_block_stride;
  const uniform<int> remaining_input_columns =
      input_vector_length - input_block_stride * full_input_blocks;

  // Loop over input_vector in blocks of input_columns_per_threadgroup
  for (int input_block_index = 0; input_block_index < full_input_blocks;
       ++input_block_index) {
    // Load vector coefficients (unchecked)
    {
      const device T* input_vector_row =
          batch_input_vector + batch_row * input_dimension;
      MTL_PRAGMA_UNROLL
      for (uint input_col_offset = 0; input_col_offset < thread_out_cols;
           input_col_offset++) {
        vector_coefficients[input_col_offset] = static_cast<float>(
            input_vector_row[input_block_col_offset + input_col_offset]
        );
      }
    }

    // Per thread work loop
    int matrix_row_offset = 0;
    MTL_PRAGMA_UNROLL
    for (uint output_row_offset = 0; output_row_offset < thread_out_rows;
         output_row_offset++) {
      // Load matrix row (unchecked)
      MTL_PRAGMA_UNROLL
      for (uint input_col_offset = 0; input_col_offset < thread_out_cols;
           input_col_offset++) {
        matrix_values[input_col_offset] = thread_matrix
            [matrix_row_offset + input_block_col_offset + input_col_offset];
      }

      // Accumulate results
      MTL_PRAGMA_UNROLL
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
    // Load vector coefficients (checked)
    {
      const device T* input_vector_row =
          batch_input_vector + batch_row * input_dimension;
      if (input_block_col_offset + int(thread_out_cols) <=
          input_vector_length) {
        MTL_PRAGMA_UNROLL
        for (uint input_col_offset = 0; input_col_offset < thread_out_cols;
             input_col_offset++) {
          vector_coefficients[input_col_offset] = static_cast<float>(
              input_vector_row[input_block_col_offset + input_col_offset]
          );
        }
      } else {
        MTL_PRAGMA_UNROLL
        for (uint input_col_offset = 0; input_col_offset < thread_out_cols;
             input_col_offset++) {
          vector_coefficients[input_col_offset] =
              input_block_col_offset + int(input_col_offset) <
                      input_vector_length
                  ? static_cast<float>(
                        input_vector_row
                            [input_block_col_offset + input_col_offset]
                    )
                  : 0.0f;
        }
      }
    }

    // Per thread work loop
    MTL_PRAGMA_UNROLL
    for (uint output_row_offset = 0; output_row_offset < thread_out_rows;
         output_row_offset++) {
      // Load matrix row (checked)
      if (input_block_col_offset + int(thread_out_cols) <=
          input_vector_length) {
        MTL_PRAGMA_UNROLL
        for (uint input_col_offset = 0; input_col_offset < thread_out_cols;
             input_col_offset++) {
          matrix_values[input_col_offset] = thread_matrix
              [output_row_offset * matrix_leading_dimension +
               input_block_col_offset + input_col_offset];
        }
      } else {
        MTL_PRAGMA_UNROLL
        for (uint input_col_offset = 0; input_col_offset < thread_out_cols;
             input_col_offset++) {
          matrix_values[input_col_offset] =
              input_block_col_offset + int(input_col_offset) <
                      input_vector_length
                  ? thread_matrix
                        [output_row_offset * matrix_leading_dimension +
                         input_block_col_offset + input_col_offset]
                  : T(0);
        }
      }

      // Accumulate results
      MTL_PRAGMA_UNROLL
      for (uint input_col_offset = 0; input_col_offset < thread_out_cols;
           input_col_offset++) {
        accumulated_values[output_row_offset] +=
            static_cast<float>(matrix_values[input_col_offset]) *
            vector_coefficients[input_col_offset];
      }
    }
  }

  // Simdgroup accumulations
  MTL_PRAGMA_UNROLL
  for (uint output_row_offset = 0; output_row_offset < thread_out_rows;
       output_row_offset++) {
    MTL_PRAGMA_UNROLL
    for (ushort simd_shuffle_offset = (sg_thread_cols / 2);
         simd_shuffle_offset >= 1;
         simd_shuffle_offset >>= 1) {
      accumulated_values[output_row_offset] += simd_shuffle_down(
          accumulated_values[output_row_offset],
          simd_shuffle_offset
      );
    }
  }

  // Threadgroup reduction (only when tg_simd_cols > 1)
  if (tg_simd_cols > 1) {
    const int computed_output_rows_per_tg =
        int(tg_simd_rows) * int(sg_thread_rows) * int(thread_out_rows);

    if (thread_col_in_simdgroup == 0) {
      threadgroup float* threadgroup_partial_accumulations =
          threadgroup_memory +
          simdgroup_column_index *
              (computed_output_rows_per_tg + int(thread_out_rows)) +
          output_block_row_offset;
      MTL_PRAGMA_UNROLL
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
          MTL_PRAGMA_UNROLL
          for (uint output_row_offset = 0; output_row_offset < thread_out_rows;
               output_row_offset++) {
            accumulated_values[output_row_offset] += base_partial
                [reduction_simdgroup_col *
                     (computed_output_rows_per_tg + int(thread_out_rows)) +
                 output_row_offset];
          }
        }
      }
    }
  }

  // Write outputs
  if (simdgroup_col_thread_base == 0 && thread_col_in_simdgroup == 0) {
    device T* output_row_values =
        batch_output_vector + batch_row * output_dimension;
    MTL_PRAGMA_UNROLL
    for (uint output_row_offset = 0; output_row_offset < thread_out_rows;
         output_row_offset++) {
      if (apply_output_scale_and_accumulate) {
        output_row_values[output_row_start + output_row_offset] =
            static_cast<T>(output_scale) *
                static_cast<T>(accumulated_values[output_row_offset]) +
            static_cast<T>(output_accumulate_scale) *
                batch_output_source
                    [(output_row_start + output_row_offset) *
                     output_source_stride];
      } else {
        output_row_values[output_row_start + output_row_offset] =
            static_cast<T>(accumulated_values[output_row_offset]);
      }
    }
  }
}
