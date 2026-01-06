// SPDX-License-Identifier: MIT
// General matrix multiplication kernels for uzu

#include <metal_stdlib>
#include "gemm.h"

using namespace metal;
using namespace uzu::matmul;

///////////////////////////////////////////////////////////////////////////////
// Function constants for compile-time specialization
///////////////////////////////////////////////////////////////////////////////

constant bool has_batch [[function_constant(10)]];

constant bool use_out_source [[function_constant(100)]];
constant bool do_axpby [[function_constant(110)]];

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

///////////////////////////////////////////////////////////////////////////////
// GEMM kernel template
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void gemm(
    const device T* input_a [[buffer(0)]],
    const device T* input_b [[buffer(1)]],
    const device T* input_c [[buffer(2), function_constant(use_out_source)]],
    device T* output [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    const constant GEMMAddMMParams* addmm_params
    [[buffer(5), function_constant(use_out_source)]],
    const constant int* batch_shape [[buffer(6), function_constant(has_batch)]],
    const constant int64_t* batch_strides
    [[buffer(7), function_constant(has_batch)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]]
) {
  // Pacifying compiler
  (void)thread_position;

  using gemm_kernel = GEMMKernel<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      transpose_a,
      transpose_b,
      true,
      true,
      AccumType>;

  using loader_a_t = typename gemm_kernel::loader_a_t;
  using loader_b_t = typename gemm_kernel::loader_b_t;
  using mma_t = typename gemm_kernel::mma_t;

  // Find block with swizzle
  const int threadgroup_y =
      ((threadgroup_position.y) << params->swizzle_log) +
      ((threadgroup_position.x) & ((1 << params->swizzle_log) - 1));
  const int threadgroup_x = (threadgroup_position.x) >> params->swizzle_log;

  // Exit early if out of bounds
  if (params->tiles_n <= threadgroup_x || params->tiles_m <= threadgroup_y) {
    return;
  }

  // Adjust for batch
  if (has_batch) {
    // Simple batch striding (no broadcast)
    input_a += params->batch_stride_a * threadgroup_position.z;
    input_b += params->batch_stride_b * threadgroup_position.z;

    if (use_out_source) {
      input_c += addmm_params->batch_stride_c * threadgroup_position.z;
    }
  } else {
    input_a += params->batch_stride_a * threadgroup_position.z;
    input_b += params->batch_stride_b * threadgroup_position.z;

    if (use_out_source) {
      input_c += addmm_params->batch_stride_c * threadgroup_position.z;
    }
  }

  output += params->batch_stride_d * threadgroup_position.z;

  // Prepare threadgroup memory
  threadgroup T threadgroup_a[gemm_kernel::tgp_mem_size_a];
  threadgroup T threadgroup_b[gemm_kernel::tgp_mem_size_b];

  threadgroup_barrier(mem_flags::mem_none);

  // Find block in input_a, input_b, input_c
  const int output_row = threadgroup_y * BM;
  const int output_col = threadgroup_x * BN;
  const size_t output_row_long = size_t(output_row);
  const size_t output_col_long = size_t(output_col);

  input_a += transpose_a ? output_row_long : output_row_long * params->lda;
  input_b += transpose_b ? output_col_long * params->ldb : output_col_long;
  output += output_row_long * params->ldd + output_col_long;

  if (use_out_source) {
    input_c += output_row_long * addmm_params->ldc +
               output_col_long * addmm_params->fdc;
  }

  // Prepare threadgroup mma operation
  thread mma_t matrix_multiply_accumulate(simd_group_id, simd_lane_id);

  // Prepare threadgroup loading operations
  thread loader_a_t block_loader_a(
      input_a,
      params->lda,
      threadgroup_a,
      simd_group_id,
      simd_lane_id
  );
  thread loader_b_t block_loader_b(
      input_b,
      params->ldb,
      threadgroup_b,
      simd_group_id,
      simd_lane_id
  );

  // Prepare threadgroup bounds
  const short threadgroup_block_batch =
      align_M ? BM : short(min(BM, params->batch - output_row));
  const short threadgroup_block_output =
      align_N ? BN : short(min(BN, params->output_dim - output_col));

  // Prepare iterations
  int gemm_k_iterations = params->gemm_k_iterations_aligned;

  // Do unaligned K iterations first
  if (!align_K) {
    const int input_aligned_end = params->gemm_k_iterations_aligned * BK;
    const int input_remainder = params->input_dim - input_aligned_end;
    const size_t input_offset_a = transpose_a
                                      ? params->lda * size_t(input_aligned_end)
                                      : size_t(input_aligned_end);
    const size_t input_offset_b = transpose_b
                                      ? size_t(input_aligned_end)
                                      : params->ldb * size_t(input_aligned_end);

    // Move loader source ahead to end
    block_loader_a.src += input_offset_a;
    block_loader_b.src += input_offset_b;

    // Load tile
    const short2 tile_dimensions_a =
        transpose_a ? short2(threadgroup_block_batch, input_remainder)
                    : short2(input_remainder, threadgroup_block_batch);
    const short2 tile_dimensions_b =
        transpose_b ? short2(input_remainder, threadgroup_block_output)
                    : short2(threadgroup_block_output, input_remainder);

    block_loader_a.load_safe(tile_dimensions_a);
    block_loader_b.load_safe(tile_dimensions_b);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Do matmul
    matrix_multiply_accumulate.mma(threadgroup_a, threadgroup_b);

    // Reset source back to start
    block_loader_a.src -= input_offset_a;
    block_loader_b.src -= input_offset_b;
  }

  const TransformAdd<AccumType, AccumType> epilogue_operation_add(
      addmm_params->alpha,
      addmm_params->beta
  );
  const TransformAxpby<AccumType, AccumType> epilogue_operation_axpby(
      addmm_params->alpha,
      addmm_params->beta
  );

  ///////////////////////////////////////////////////////////////////////////
  // MNK aligned loop
  if (align_M && align_N) {
    // Do gemm
    for (int iteration = 0; iteration < gemm_k_iterations; iteration++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // Load elements into threadgroup
      block_loader_a.load_unsafe();
      block_loader_b.load_unsafe();

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Multiply and accumulate threadgroup elements
      matrix_multiply_accumulate.mma(threadgroup_a, threadgroup_b);

      // Prepare for next iteration
      block_loader_a.next();
      block_loader_b.next();
    }

    threadgroup_barrier(mem_flags::mem_none);

    // Do epilogue
    if (use_out_source) {
      if (do_axpby) {
        matrix_multiply_accumulate.apply_epilogue(
            input_c,
            addmm_params->ldc,
            addmm_params->fdc,
            epilogue_operation_axpby
        );
      } else {
        matrix_multiply_accumulate.apply_epilogue(
            input_c,
            addmm_params->ldc,
            addmm_params->fdc,
            epilogue_operation_add
        );
      }
    }

    // Store results to device memory
    return matrix_multiply_accumulate.store_result(output, params->ldd);
  }
  ///////////////////////////////////////////////////////////////////////////
  // MN unaligned loop
  else { // Loop over K - unaligned case
    const int leftover_block_input = 0;

    if ((align_M || threadgroup_block_batch == BM) &&
        (align_N || threadgroup_block_output == BN)) {
      // Do gemm
      gemm_kernel::gemm_loop(
          threadgroup_a,
          threadgroup_b,
          gemm_k_iterations,
          block_loader_a,
          block_loader_b,
          matrix_multiply_accumulate,
          threadgroup_block_batch,
          threadgroup_block_output,
          leftover_block_input,
          LoopAlignment<true, true, true>{}
      );

      // Do epilogue
      if (use_out_source) {
        if (do_axpby) {
          matrix_multiply_accumulate.apply_epilogue(
              input_c,
              addmm_params->ldc,
              addmm_params->fdc,
              epilogue_operation_axpby
          );
        } else {
          matrix_multiply_accumulate.apply_epilogue(
              input_c,
              addmm_params->ldc,
              addmm_params->fdc,
              epilogue_operation_add
          );
        }
      }

      // Store results to device memory
      return matrix_multiply_accumulate.store_result(output, params->ldd);

    } else if (align_N || threadgroup_block_output == BN) {
      gemm_kernel::gemm_loop(
          threadgroup_a,
          threadgroup_b,
          gemm_k_iterations,
          block_loader_a,
          block_loader_b,
          matrix_multiply_accumulate,
          threadgroup_block_batch,
          threadgroup_block_output,
          leftover_block_input,
          LoopAlignment<false, true, true>{}
      );

      // Do epilogue
      if (use_out_source) {
        if (do_axpby) {
          matrix_multiply_accumulate.apply_epilogue_safe(
              input_c,
              addmm_params->ldc,
              addmm_params->fdc,
              short2(threadgroup_block_output, threadgroup_block_batch),
              epilogue_operation_axpby
          );
        } else {
          matrix_multiply_accumulate.apply_epilogue_safe(
              input_c,
              addmm_params->ldc,
              addmm_params->fdc,
              short2(threadgroup_block_output, threadgroup_block_batch),
              epilogue_operation_add
          );
        }
      }

      // Store results to device memory
      return matrix_multiply_accumulate.store_result_safe(
          output,
          params->ldd,
          short2(threadgroup_block_output, threadgroup_block_batch)
      );

    } else if (align_M || threadgroup_block_batch == BM) {
      gemm_kernel::gemm_loop(
          threadgroup_a,
          threadgroup_b,
          gemm_k_iterations,
          block_loader_a,
          block_loader_b,
          matrix_multiply_accumulate,
          threadgroup_block_batch,
          threadgroup_block_output,
          leftover_block_input,
          LoopAlignment<true, false, true>{}
      );

      // Do epilogue
      if (use_out_source) {
        if (do_axpby) {
          matrix_multiply_accumulate.apply_epilogue_safe(
              input_c,
              addmm_params->ldc,
              addmm_params->fdc,
              short2(threadgroup_block_output, threadgroup_block_batch),
              epilogue_operation_axpby
          );
        } else {
          matrix_multiply_accumulate.apply_epilogue_safe(
              input_c,
              addmm_params->ldc,
              addmm_params->fdc,
              short2(threadgroup_block_output, threadgroup_block_batch),
              epilogue_operation_add
          );
        }
      }

      // Store results to device memory
      return matrix_multiply_accumulate.store_result_safe(
          output,
          params->ldd,
          short2(threadgroup_block_output, threadgroup_block_batch)
      );

    } else {
      gemm_kernel::gemm_loop(
          threadgroup_a,
          threadgroup_b,
          gemm_k_iterations,
          block_loader_a,
          block_loader_b,
          matrix_multiply_accumulate,
          threadgroup_block_batch,
          threadgroup_block_output,
          leftover_block_input,
          LoopAlignment<false, false, true>{}
      );

      // Do epilogue
      if (use_out_source) {
        if (do_axpby) {
          matrix_multiply_accumulate.apply_epilogue_safe(
              input_c,
              addmm_params->ldc,
              addmm_params->fdc,
              short2(threadgroup_block_output, threadgroup_block_batch),
              epilogue_operation_axpby
          );
        } else {
          matrix_multiply_accumulate.apply_epilogue_safe(
              input_c,
              addmm_params->ldc,
              addmm_params->fdc,
              short2(threadgroup_block_output, threadgroup_block_batch),
              epilogue_operation_add
          );
        }
      }

      // Store results to device memory
      return matrix_multiply_accumulate.store_result_safe(
          output,
          params->ldd,
          short2(threadgroup_block_output, threadgroup_block_batch)
      );
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Kernel instantiation macros
///////////////////////////////////////////////////////////////////////////////

#define instantiate_gemm(                                                      \
    transpose_name,                                                            \
    transpose_a,                                                               \
    transpose_b,                                                               \
    type_name,                                                                 \
    element_type,                                                              \
    block_batch,                                                               \
    block_output,                                                              \
    block_input,                                                               \
    warp_batch,                                                                \
    warp_output                                                                \
)                                                                              \
  template [[host_name(                                                        \
      "gemm_" #transpose_name "_" #type_name "_bm" #block_batch                \
      "_bn" #block_output "_bk" #block_input "_wm" #warp_batch                 \
      "_wn" #warp_output                                                       \
  )]] [[kernel]] void                                                          \
  gemm<                                                                        \
      element_type,                                                            \
      block_batch,                                                             \
      block_output,                                                            \
      block_input,                                                             \
      warp_batch,                                                              \
      warp_output,                                                             \
      transpose_a,                                                             \
      transpose_b,                                                             \
      float>(                                                                  \
      const device element_type* input_a [[buffer(0)]],                        \
      const device element_type* input_b [[buffer(1)]],                        \
      const device element_type* input_c                                       \
      [[buffer(2), function_constant(use_out_source)]],                        \
      device element_type* output [[buffer(3)]],                               \
      const constant GEMMParams* params [[buffer(4)]],                         \
      const constant GEMMAddMMParams* addmm_params                             \
      [[buffer(5), function_constant(use_out_source)]],                        \
      const constant int* batch_shape                                          \
      [[buffer(6), function_constant(has_batch)]],                             \
      const constant int64_t* batch_strides                                    \
      [[buffer(7), function_constant(has_batch)]],                             \
      uint simd_lane_id [[thread_index_in_simdgroup]],                         \
      uint simd_group_id [[simdgroup_index_in_threadgroup]],                   \
      uint3 threadgroup_position [[threadgroup_position_in_grid]],             \
      uint3 thread_position [[thread_position_in_threadgroup]]                 \
  );

// clang-format off
#define instantiate_gemm_transpose_helper(                                                       \
    type_name, element_type, block_batch, block_output, block_input, warp_batch, warp_output     \
)                                                                                                \
  instantiate_gemm(nn, false, false, type_name, element_type, block_batch, block_output,         \
                   block_input, warp_batch, warp_output)                                         \
  instantiate_gemm(nt, false, true, type_name, element_type, block_batch, block_output,          \
                   block_input, warp_batch, warp_output)                                         \
  instantiate_gemm(tn, true, false, type_name, element_type, block_batch, block_output,          \
                   block_input, warp_batch, warp_output)                                         \
  instantiate_gemm(tt, true, true, type_name, element_type, block_batch, block_output,           \
                   block_input, warp_batch, warp_output)
// clang-format on

// clang-format off
#define instantiate_gemm_shapes_helper(type_name, element_type)                 \
  instantiate_gemm_transpose_helper(type_name, element_type, 64, 64, 16, 1, 2)  \
  instantiate_gemm_transpose_helper(type_name, element_type, 64, 32, 32, 2, 2)  \
  instantiate_gemm_transpose_helper(type_name, element_type, 32, 64, 16, 1, 2)  \
  instantiate_gemm_transpose_helper(type_name, element_type, 32, 32, 16, 2, 2)  \
  /* NAX-oriented larger tiles (safe BK=32) */                                  \
  instantiate_gemm_transpose_helper(type_name, element_type, 64, 64, 32, 2, 2)  \
  instantiate_gemm_transpose_helper(type_name, element_type, 128, 128, 32, 2, 2)
// clang-format on

// Instantiate for float16, bfloat16, and float32
instantiate_gemm_shapes_helper(f16, half);
instantiate_gemm_shapes_helper(bf16, bfloat);
instantiate_gemm_shapes_helper(f32, float);

///////////////////////////////////////////////////////////////////////////////
// Simple GEMM kernel (for smaller matrices or vector operations)
///////////////////////////////////////////////////////////////////////////////

template <typename T>
[[kernel]] void gemv(
    const device T* input_a [[buffer(0)]],
    const device T* input_b [[buffer(1)]],
    device T* output [[buffer(2)]],
    constant int& batch [[buffer(3)]],
    constant int& output_dim [[buffer(4)]],
    constant int& input_dim [[buffer(5)]],
    constant int& leading_dimension_a [[buffer(6)]],
    constant int& leading_dimension_b [[buffer(7)]],
    constant int& leading_dimension_output [[buffer(8)]],
    uint2 thread_position [[thread_position_in_grid]],
    uint2 grid_size [[threads_per_grid]]
) {
  const int row_index = thread_position.y;
  const int column_index = thread_position.x;

  if (row_index >= batch || column_index >= output_dim) {
    return;
  }

  float accumulator = 0.0f;
  for (int input_index = 0; input_index < input_dim; input_index++) {
    accumulator +=
        float(input_a[row_index * leading_dimension_a + input_index]) *
        float(input_b[input_index * leading_dimension_b + column_index]);
  }

  output[row_index * leading_dimension_output + column_index] = T(accumulator);
}

// Simple GEMV instantiations
template [[host_name("gemv_f16")]] [[kernel]] void gemv<half>(
    const device half* input_a [[buffer(0)]],
    const device half* input_b [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant int& batch [[buffer(3)]],
    constant int& output_dim [[buffer(4)]],
    constant int& input_dim [[buffer(5)]],
    constant int& leading_dimension_a [[buffer(6)]],
    constant int& leading_dimension_b [[buffer(7)]],
    constant int& leading_dimension_output [[buffer(8)]],
    uint2 thread_position [[thread_position_in_grid]],
    uint2 grid_size [[threads_per_grid]]
);

template [[host_name("gemv_bf16")]] [[kernel]] void gemv<bfloat>(
    const device bfloat* input_a [[buffer(0)]],
    const device bfloat* input_b [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant int& batch [[buffer(3)]],
    constant int& output_dim [[buffer(4)]],
    constant int& input_dim [[buffer(5)]],
    constant int& leading_dimension_a [[buffer(6)]],
    constant int& leading_dimension_b [[buffer(7)]],
    constant int& leading_dimension_output [[buffer(8)]],
    uint2 thread_position [[thread_position_in_grid]],
    uint2 grid_size [[threads_per_grid]]
);

template [[host_name("gemv_f32")]] [[kernel]] void gemv<float>(
    const device float* input_a [[buffer(0)]],
    const device float* input_b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& batch [[buffer(3)]],
    constant int& output_dim [[buffer(4)]],
    constant int& input_dim [[buffer(5)]],
    constant int& leading_dimension_a [[buffer(6)]],
    constant int& leading_dimension_b [[buffer(7)]],
    constant int& leading_dimension_output [[buffer(8)]],
    uint2 thread_position [[thread_position_in_grid]],
    uint2 grid_size [[threads_per_grid]]
);
