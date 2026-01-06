// Split-K GEMM: partitions K dimension across threadgroups for better
// parallelism on small M*N with large K.

#include <metal_stdlib>
#include "gemm.h"

using namespace metal;
using namespace uzu::matmul;

///////////////////////////////////////////////////////////////////////////////
// Split-K Partial GEMM Kernel
///////////////////////////////////////////////////////////////////////////////

template <
    typename InputType,
    typename AccumulatorType,
    int TileRows,
    int TileCols,
    int TileDepth,
    int WarpsPerRow,
    int WarpsPerCol,
    bool TransposeA,
    bool TransposeB,
    bool MNAligned,
    bool KAligned>
[[kernel, max_total_threads_per_threadgroup(WarpsPerRow * WarpsPerCol * 32)]]
void splitk_partial_gemm(
    const device InputType* input_a [[buffer(0)]],
    const device InputType* input_b [[buffer(1)]],
    device AccumulatorType* accumulator [[buffer(2)]],
    const constant SplitKGEMMParams* params [[buffer(3)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]]) {
  (void)thread_id;

  using gemm_kernel = GEMMKernel<
      InputType,
      AccumulatorType,
      TileRows,
      TileCols,
      TileDepth,
      WarpsPerRow,
      WarpsPerCol,
      TransposeA,
      TransposeB,
      MNAligned,
      KAligned,
      AccumulatorType>;

  using loader_a_t = typename gemm_kernel::loader_a_t;
  using loader_b_t = typename gemm_kernel::loader_b_t;
  using mma_t = typename gemm_kernel::mma_t;

  threadgroup InputType shared_a[gemm_kernel::tgp_mem_size_a];
  threadgroup InputType shared_b[gemm_kernel::tgp_mem_size_b];

  const int tile_x = threadgroup_id.x;
  const int tile_y = threadgroup_id.y;
  const int partition_index = threadgroup_id.z;

  if (params->tile_count_n <= tile_x || params->tile_count_m <= tile_y) {
    return;
  }

  const int output_row = tile_y * TileRows;
  const int output_col = tile_x * TileCols;
  const int k_partition_start = params->k_elements_per_partition * partition_index;

  const size_t output_row_long = size_t(output_row);
  const size_t output_col_long = size_t(output_col);
  const size_t k_start_long = size_t(k_partition_start);

  const device InputType* a_ptr = input_a;
  const device InputType* b_ptr = input_b;

  a_ptr += TransposeA
      ? (output_row_long + k_start_long * params->leading_dim_a)
      : (k_start_long + output_row_long * params->leading_dim_a);
  b_ptr += TransposeB
      ? (k_start_long + output_col_long * params->leading_dim_b)
      : (output_col_long + k_start_long * params->leading_dim_b);

  device AccumulatorType* output_ptr = accumulator +
      (size_t(params->output_elements_per_partition) * partition_index) +
      (output_row_long * params->leading_dim_accumulator + output_col_long);

  thread loader_a_t loader_a(a_ptr, params->leading_dim_a, shared_a, simd_group_id, simd_lane_id);
  thread loader_b_t loader_b(b_ptr, params->leading_dim_b, shared_b, simd_group_id, simd_lane_id);
  thread mma_t mma_op(simd_group_id, simd_lane_id);

  int gemm_k_iterations = params->gemm_k_iterations_aligned;

  short tile_bound_m = min(TileRows, params->m - output_row);
  short tile_bound_n = min(TileCols, params->n - output_col);
  short leftover_k = params->k % TileDepth;

  const bool is_last_partition = (partition_index + 1) == params->partition_count;

  if (MNAligned || (tile_bound_m == TileRows && tile_bound_n == TileCols)) {
    gemm_kernel::gemm_loop(
        shared_a, shared_b,
        gemm_k_iterations,
        loader_a, loader_b, mma_op,
        tile_bound_m, tile_bound_n, leftover_k,
        LoopAlignment<true, true, true>{});
  } else if (tile_bound_n == TileCols) {
    gemm_kernel::gemm_loop(
        shared_a, shared_b,
        gemm_k_iterations,
        loader_a, loader_b, mma_op,
        tile_bound_m, tile_bound_n, leftover_k,
        LoopAlignment<false, true, true>{});
  } else if (tile_bound_m == TileRows) {
    gemm_kernel::gemm_loop(
        shared_a, shared_b,
        gemm_k_iterations,
        loader_a, loader_b, mma_op,
        tile_bound_m, tile_bound_n, leftover_k,
        LoopAlignment<true, false, true>{});
  } else {
    gemm_kernel::gemm_loop(
        shared_a, shared_b,
        gemm_k_iterations,
        loader_a, loader_b, mma_op,
        tile_bound_m, tile_bound_n, leftover_k,
        LoopAlignment<false, false, true>{});
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (is_last_partition) {
    int remaining_k_iterations =
        (params->k - (k_partition_start + params->k_elements_per_partition)) / TileDepth;
    if (!KAligned || remaining_k_iterations > 0) {
      gemm_kernel::gemm_loop(
          shared_a, shared_b,
          remaining_k_iterations,
          loader_a, loader_b, mma_op,
          tile_bound_m, tile_bound_n, leftover_k,
          LoopAlignment<false, false, KAligned>{});
    }
  }

  if (MNAligned || (tile_bound_m == TileRows && tile_bound_n == TileCols)) {
    mma_op.store_result(output_ptr, params->leading_dim_accumulator);
  } else {
    mma_op.store_result_safe(output_ptr, params->leading_dim_accumulator, short2(tile_bound_n, tile_bound_m));
  }
}

///////////////////////////////////////////////////////////////////////////////
// Split-K Accumulation Kernel
///////////////////////////////////////////////////////////////////////////////

template <typename OutputType, typename AccumulatorType>
[[kernel]] void splitk_accumulate(
    const device AccumulatorType* partitioned_accumulator [[buffer(0)]],
    device OutputType* output [[buffer(1)]],
    constant int& partition_count [[buffer(2)]],
    constant int& output_elements_per_partition [[buffer(3)]],
    constant int& leading_dim_output [[buffer(4)]],
    uint2 thread_position [[thread_position_in_grid]]) {
  const size_t col = thread_position.x;
  const size_t row = thread_position.y;

  output += col + row * size_t(leading_dim_output);
  partitioned_accumulator += col + row * size_t(leading_dim_output);

  AccumulatorType sum = AccumulatorType(0);
  size_t partition_offset = 0;

  for (int i = 0; i < partition_count; i++) {
    sum += partitioned_accumulator[partition_offset];
    partition_offset += output_elements_per_partition;
  }

  output[0] = OutputType(sum);
}

///////////////////////////////////////////////////////////////////////////////
// Kernel Instantiation
///////////////////////////////////////////////////////////////////////////////

#define INSTANTIATE_SPLITK_PARTIAL(                                            \
    transpose_name, transpose_a, transpose_b,                                  \
    type_name, input_type,                                                     \
    tile_rows, tile_cols, tile_depth, warps_row, warps_col)                    \
  template [[host_name(                                                        \
      "splitk_partial_" #transpose_name "_" #type_name                         \
      "_tm" #tile_rows "_tn" #tile_cols "_tk" #tile_depth                      \
      "_wm" #warps_row "_wn" #warps_col                                        \
  )]] [[kernel]] void                                                          \
  splitk_partial_gemm<                                                         \
      input_type, float,                                                       \
      tile_rows, tile_cols, tile_depth, warps_row, warps_col,                  \
      transpose_a, transpose_b, false, false>(                                 \
      const device input_type* input_a [[buffer(0)]],                          \
      const device input_type* input_b [[buffer(1)]],                          \
      device float* accumulator [[buffer(2)]],                                 \
      const constant SplitKGEMMParams* params [[buffer(3)]],                   \
      uint simd_lane_id [[thread_index_in_simdgroup]],                         \
      uint simd_group_id [[simdgroup_index_in_threadgroup]],                   \
      uint3 threadgroup_id [[threadgroup_position_in_grid]],                   \
      uint3 thread_id [[thread_position_in_threadgroup]]);

#define INSTANTIATE_SPLITK_TRANSPOSE_HELPER(                                   \
    type_name, input_type, tile_rows, tile_cols, tile_depth, warps_row, warps_col) \
  INSTANTIATE_SPLITK_PARTIAL(nn, false, false, type_name, input_type,          \
      tile_rows, tile_cols, tile_depth, warps_row, warps_col)                  \
  INSTANTIATE_SPLITK_PARTIAL(nt, false, true, type_name, input_type,           \
      tile_rows, tile_cols, tile_depth, warps_row, warps_col)                  \
  INSTANTIATE_SPLITK_PARTIAL(tn, true, false, type_name, input_type,           \
      tile_rows, tile_cols, tile_depth, warps_row, warps_col)                  \
  INSTANTIATE_SPLITK_PARTIAL(tt, true, true, type_name, input_type,            \
      tile_rows, tile_cols, tile_depth, warps_row, warps_col)

#define INSTANTIATE_SPLITK_SHAPES_HELPER(type_name, input_type)                \
  INSTANTIATE_SPLITK_TRANSPOSE_HELPER(type_name, input_type, 16, 16, 16, 2, 2) \
  INSTANTIATE_SPLITK_TRANSPOSE_HELPER(type_name, input_type, 16, 32, 16, 2, 2) \
  INSTANTIATE_SPLITK_TRANSPOSE_HELPER(type_name, input_type, 32, 16, 16, 2, 2) \
  INSTANTIATE_SPLITK_TRANSPOSE_HELPER(type_name, input_type, 32, 32, 16, 2, 2)

INSTANTIATE_SPLITK_SHAPES_HELPER(f16, half)
INSTANTIATE_SPLITK_SHAPES_HELPER(bf16, bfloat)
INSTANTIATE_SPLITK_SHAPES_HELPER(f32, float)

#define INSTANTIATE_SPLITK_ACCUM(type_name, output_type)                       \
  template [[host_name("splitk_accum_" #type_name)]] [[kernel]] void           \
  splitk_accumulate<output_type, float>(                                       \
      const device float* partitioned_accumulator [[buffer(0)]],               \
      device output_type* output [[buffer(1)]],                                \
      constant int& partition_count [[buffer(2)]],                             \
      constant int& output_elements_per_partition [[buffer(3)]],               \
      constant int& leading_dim_output [[buffer(4)]],                          \
      uint2 thread_position [[thread_position_in_grid]]);

INSTANTIATE_SPLITK_ACCUM(f16, half)
INSTANTIATE_SPLITK_ACCUM(bf16, bfloat)
INSTANTIATE_SPLITK_ACCUM(f32, float)
