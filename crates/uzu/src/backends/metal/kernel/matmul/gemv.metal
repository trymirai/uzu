#include <metal_stdlib>

#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"

#include <metal_simdgroup>

using namespace metal;

// Upper bound for threadgroup memory (in floats).
// Max config: tg_simd_cols=16, output_rows_per_tg=16, thread_out_rows=4
// → 16 * (16 + 4) = 320. Unused when tg_simd_cols == 1.
#define GEMV_MAX_THREADGROUP_MEMORY 320

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MatmulGemv)(
    const device T* matrix,
    const device T* input_vector,
    const device T* output_bias OPTIONAL(is_bias),
    device T* output_vector,
    const constant uint& input_dimension,
    const constant uint& output_dimension,
    const constant uint& matrix_leading_dimension,
    const constant float& output_scale,
    const constant uint& batch_rows,
    const constant uint& output_rows_per_threadgroup,
    threadgroup float threadgroup_memory[GEMV_MAX_THREADGROUP_MEMORY],
    const uint tg_simd_rows SPECIALIZE,
    const uint tg_simd_cols SPECIALIZE,
    const uint sg_thread_rows SPECIALIZE,
    const uint sg_thread_cols SPECIALIZE,
    const uint thread_out_rows SPECIALIZE,
    const uint thread_out_cols SPECIALIZE,
    const bool is_accumulate SPECIALIZE,
    const bool is_bias SPECIALIZE,
    const uint threadgroup_index_x GROUPS((output_dimension + output_rows_per_threadgroup - 1) / output_rows_per_threadgroup),
    const uint threadgroup_index_y GROUPS(batch_rows),
    const uint thread_index_x THREADS(32),
    const uint thread_index_y THREADS(16),
    const uint thread_index_z THREADS(1),
    const ThreadContext thread_context
) {
  if (output_rows_per_threadgroup <= 0) {
    return;
  }

  // Simdgroup guard — kill excess simdgroups beyond what this tile uses
  if (thread_context.threadgroup_index >= tg_simd_rows * tg_simd_cols) {
    return;
  }

  // Derived constants from SPECIALIZE params
  const uint threads_per_threadgroup_col = tg_simd_cols * sg_thread_cols;
  const uint input_columns_per_threadgroup =
      threads_per_threadgroup_col * thread_out_cols;

  const uint batch_row = static_cast<uint>(threadgroup_index_y);
  if (batch_row >= batch_rows) {
    return;
  }

  // Thread local accumulation results
  thread float accumulated_values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  thread T matrix_values[4];
  thread float vector_coefficients[4];

  const uint thread_row_in_simdgroup =
      sg_thread_cols != 32
          ? uint(thread_context.simdgroup_index) / uint(sg_thread_cols)
          : 0;
  const int thread_col_in_simdgroup =
      sg_thread_cols != 32
          ? uint(thread_context.simdgroup_index) % uint(sg_thread_cols)
          : uint(thread_context.simdgroup_index);

  const int simdgroup_column_index =
      tg_simd_cols != 1 ? uint(thread_context.threadgroup_index % tg_simd_cols)
                        : 0;

  const uint simdgroup_row_thread_base =
      tg_simd_cols != 1
          ? uint(sg_thread_rows) *
                uint(thread_context.threadgroup_index / tg_simd_cols)
          : uint(sg_thread_rows) * uint(thread_context.threadgroup_index);
  const uint simdgroup_col_thread_base =
      tg_simd_cols != 1
          ? uint(sg_thread_cols) *
                uint(thread_context.threadgroup_index % tg_simd_cols)
          : 0;

  uint output_block_row_offset =
      (simdgroup_row_thread_base + thread_row_in_simdgroup) *
      uint(thread_out_rows);
  uint input_block_col_offset =
      (simdgroup_col_thread_base + thread_col_in_simdgroup) *
      uint(thread_out_cols);

  // Block position
  uint output_row_start =
      uint(threadgroup_index_x) * uint(output_rows_per_threadgroup) +
      output_block_row_offset;

  // Exit simdgroup if rows out of bound
  if (output_row_start >= output_dimension)
    return;

  // Adjust tail simdgroup to ensure in bound reads
  output_row_start =
      output_row_start + uint(thread_out_rows) <= output_dimension
          ? output_row_start
          : output_dimension - uint(thread_out_rows);

  const device T* thread_matrix =
      matrix + output_row_start * matrix_leading_dimension;

  const uniform<uint> input_block_stride =
      make_uniform(uint(input_columns_per_threadgroup));
  const uniform<uint> input_vector_length = make_uniform(input_dimension);
  const uniform<uint> full_input_blocks =
      input_vector_length / input_block_stride;
  const uniform<uint> remaining_input_columns =
      input_vector_length - input_block_stride * full_input_blocks;

  // Loop over input_vector in blocks of input_columns_per_threadgroup
  for (uint input_block_index = 0; input_block_index < full_input_blocks;
       ++input_block_index) {
    // Load vector coefficients (unchecked)
    {
      const device T* input_vector_row =
          input_vector + batch_row * input_dimension;
      METAL_PRAGMA_UNROLL
      for (uint input_col_offset = 0; input_col_offset < thread_out_cols;
           input_col_offset++) {
        vector_coefficients[input_col_offset] = static_cast<float>(
            input_vector_row[input_block_col_offset + input_col_offset]
        );
      }
    }

    // Per thread work loop
    uint matrix_row_offset = 0;
    METAL_PRAGMA_UNROLL
    for (uint output_row_offset = 0; output_row_offset < thread_out_rows;
         output_row_offset++) {
      // Load matrix row (unchecked)
      METAL_PRAGMA_UNROLL
      for (uint input_col_offset = 0; input_col_offset < thread_out_cols;
           input_col_offset++) {
        matrix_values[input_col_offset] = thread_matrix
            [matrix_row_offset + input_block_col_offset + input_col_offset];
      }

      // Accumulate results
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
    // Load vector coefficients (checked)
    {
      const device T* input_vector_row =
          input_vector + batch_row * input_dimension;
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

    // Per thread work loop
    METAL_PRAGMA_UNROLL
    for (uint output_row_offset = 0; output_row_offset < thread_out_rows;
         output_row_offset++) {
      // Load matrix row (checked)
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

      // Accumulate results
      METAL_PRAGMA_UNROLL
      for (uint input_col_offset = 0; input_col_offset < thread_out_cols;
           input_col_offset++) {
        accumulated_values[output_row_offset] +=
            static_cast<float>(matrix_values[input_col_offset]) *
            vector_coefficients[input_col_offset];
      }
    }
  }

  // Simdgroup accumulations
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

  // Threadgroup reduction (only when tg_simd_cols > 1)
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
          for (uint output_row_offset = 0; output_row_offset < thread_out_rows;
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

  // Write outputs
  if (simdgroup_col_thread_base == 0 && thread_col_in_simdgroup == 0) {
    device T* output_row_values = output_vector + batch_row * output_dimension;
    METAL_PRAGMA_UNROLL
    for (uint output_row_offset = 0; output_row_offset < thread_out_rows;
         output_row_offset++) {
      T accumulated_c = static_cast<T>(output_scale) *
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
}
