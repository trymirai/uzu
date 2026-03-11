// clang-format off
#include "../../../common/utils.h"
#include "../../../definitions.metal"

#include "../../common/params.h"
#include "../../common/mpp_cooperative_matmul.h"
#include "../../common/loader.h"

using namespace uzu::matmul;

///////////////////////////////////////////////////////////////////////////////
// Threadgroup staging constants (T is #define'd by the DSL wrapper)
///////////////////////////////////////////////////////////////////////////////

#define PREFETCH_K_SIZE (((26624 / (128 * 2 * (int)sizeof(T))) / 16) * 16)
#define THREADGROUP_PADDING ((short)(16 / sizeof(T)))
#define THREADGROUP_LEADING_DIMENSION ((short)(PREFETCH_K_SIZE + THREADGROUP_PADDING))
#define THREADGROUP_TILE_SIZE (128 * THREADGROUP_LEADING_DIMENSION)

///////////////////////////////////////////////////////////////////////////////
// MPP GEMM implementation -- templated over tile config
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    short BLOCK_M,
    short BLOCK_N,
    short BLOCK_K,
    short WARPS_M,
    short WARPS_N>
METAL_FUNC void gemm_mpp_impl(
    const device T* left_matrix,
    const device T* right_matrix,
    device T* output_matrix,
    const constant GEMMParams* params,
    const bool align_m,
    const bool align_n,
    const bool align_k,
    threadgroup T* left_shared,
    threadgroup T* right_shared,
    uint simd_group_id,
    uint simd_lane_id,
    uint3 threadgroup_position
) {
  using AccumulatorType = float;

  constexpr short SUBTILE_ROWS = 16;
  constexpr short SUBTILE_COLS = 32;
  constexpr short MATMUL_K_STEP = 16;
  constexpr short SIMDGROUP_M = BLOCK_M / WARPS_M;
  constexpr short SIMDGROUP_N = BLOCK_N / WARPS_N;
  constexpr short TILES_M = SIMDGROUP_M / SUBTILE_ROWS;
  constexpr short TILES_N = SIMDGROUP_N / SUBTILE_COLS;
  constexpr short THREADGROUP_SIZE = WARPS_M * WARPS_N * 32;

  constexpr short PREFETCH_K = short(PREFETCH_K_SIZE);
  constexpr short THREADGROUP_LD = PREFETCH_K + short(16 / sizeof(T));
  constexpr short INNER_K_STEPS = PREFETCH_K / MATMUL_K_STEP;

  const int swizzle_size = 1 << params->swizzle_log;
  const int tid_y = ((threadgroup_position.y) * swizzle_size) +
                    ((threadgroup_position.x) % swizzle_size);
  const int tid_x = (threadgroup_position.x) / swizzle_size;

  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  left_matrix += params->batch_stride_a * threadgroup_position.z;
  right_matrix += params->batch_stride_b * threadgroup_position.z;
  output_matrix += params->batch_stride_d * threadgroup_position.z;

  threadgroup_barrier(mem_flags::mem_none);

  const int block_row_start = tid_y * BLOCK_M;
  const int block_col_start = tid_x * BLOCK_N;
  const size_t block_row_start_long = size_t(block_row_start);
  const size_t block_col_start_long = size_t(block_col_start);

  const device T* left_block_ptr = left_matrix + block_row_start_long * params->leading_dim_a;
  const device T* right_block_ptr = right_matrix + block_col_start_long * params->leading_dim_b;

  const short tile_row_offset = SIMDGROUP_M * (simd_group_id / WARPS_N);
  const short tile_col_offset = SIMDGROUP_N * (simd_group_id % WARPS_N);

  device T* output_ptr = output_matrix + block_row_start_long * params->leading_dim_d + block_col_start_long + tile_row_offset * params->leading_dim_d + tile_col_offset;

  const int simdgroup_limit_m_int =
      align_m ? int(SIMDGROUP_M) : min(int(SIMDGROUP_M), params->M - (block_row_start + tile_row_offset));
  const short simdgroup_limit_m = short(simdgroup_limit_m_int);

  const int simdgroup_limit_n_int =
      align_n ? int(SIMDGROUP_N) : min(int(SIMDGROUP_N), params->N - (block_col_start + tile_col_offset));
  const short simdgroup_limit_n = short(simdgroup_limit_n_int);

  const bool is_unaligned_m = align_m ? false : (simdgroup_limit_m != SIMDGROUP_M);
  const bool is_unaligned_n = align_n ? false : (simdgroup_limit_n != SIMDGROUP_N);

  // --- Fallback: direct device reads when staging gives no benefit ---
  if constexpr (PREFETCH_K <= MATMUL_K_STEP) {
    const device T* left_simdgroup_ptr = left_block_ptr + tile_row_offset * params->leading_dim_a;
    const device T* right_simdgroup_ptr = right_block_ptr + tile_col_offset * params->leading_dim_b;

    PRAGMA_UNROLL
    for (short tile_m = 0; tile_m < TILES_M; tile_m++) {
      PRAGMA_UNROLL
      for (short tile_n = 0; tile_n < TILES_N; tile_n++) {
        const short row_offset = tile_m * SUBTILE_ROWS;
        const short col_offset = tile_n * SUBTILE_COLS;

        const short m_limit = is_unaligned_m ? short(max(0, int(simdgroup_limit_m) - row_offset)) : SUBTILE_ROWS;
        const short n_limit = is_unaligned_n ? short(max(0, int(simdgroup_limit_n) - col_offset)) : SUBTILE_COLS;
        if (m_limit <= 0 || n_limit <= 0) continue;

        const device T* left_subtile_ptr = left_simdgroup_ptr + row_offset * params->leading_dim_a;
        const device T* right_subtile_ptr = right_simdgroup_ptr + col_offset * params->leading_dim_b;
        device T* output_subtile_ptr = output_ptr + row_offset * params->leading_dim_d + col_offset;

        const bool subtile_aligned_m = !is_unaligned_m || (m_limit == SUBTILE_ROWS);
        const bool subtile_aligned_n = !is_unaligned_n || (n_limit == SUBTILE_COLS);

        if (subtile_aligned_m && subtile_aligned_n && align_k) {
          cooperative_tensor_gemm<SUBTILE_ROWS, SUBTILE_COLS, MATMUL_K_STEP, AccumulatorType, T, T, T,
                                 false, true, true, true, true>(
              left_subtile_ptr, params->leading_dim_a, right_subtile_ptr, params->leading_dim_b, output_subtile_ptr, params->leading_dim_d,
              params->K, m_limit, n_limit);
        } else if (subtile_aligned_m && subtile_aligned_n) {
          cooperative_tensor_gemm<SUBTILE_ROWS, SUBTILE_COLS, MATMUL_K_STEP, AccumulatorType, T, T, T,
                                 false, true, true, true, false>(
              left_subtile_ptr, params->leading_dim_a, right_subtile_ptr, params->leading_dim_b, output_subtile_ptr, params->leading_dim_d,
              params->K, m_limit, n_limit);
        } else {
          cooperative_tensor_gemm<SUBTILE_ROWS, SUBTILE_COLS, MATMUL_K_STEP, AccumulatorType, T, T, T,
                                 false, true, false, false, false>(
              left_subtile_ptr, params->leading_dim_a, right_subtile_ptr, params->leading_dim_b, output_subtile_ptr, params->leading_dim_d,
              params->K, m_limit, n_limit);
        }
      }
    }
    return;
  }

  // --- Staged path: threadgroup memory prefetch ---

  BlockLoader<T, BLOCK_M, PREFETCH_K, THREADGROUP_LD, 1, THREADGROUP_SIZE> loader_a(
      left_block_ptr, params->leading_dim_a, left_shared, ushort(simd_group_id), ushort(simd_lane_id));
  BlockLoader<T, BLOCK_N, PREFETCH_K, THREADGROUP_LD, 1, THREADGROUP_SIZE> loader_b(
      right_block_ptr, params->leading_dim_b, right_shared, ushort(simd_group_id), ushort(simd_lane_id));

  constexpr auto matmul_descriptor = mpp::tensor_ops::matmul2d_descriptor(
      SUBTILE_ROWS, SUBTILE_COLS, MATMUL_K_STEP,
      false, true, false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

  mpp::tensor_ops::matmul2d<matmul_descriptor, metal::execution_simdgroup> matmul_operation;

  auto left_tensor = matmul_operation.template get_left_input_cooperative_tensor<T, T, AccumulatorType>();
  auto right_tensor = matmul_operation.template get_right_input_cooperative_tensor<T, T, AccumulatorType>();
  auto accumulator_tensor = matmul_operation.template get_destination_cooperative_tensor<
      decltype(left_tensor), decltype(right_tensor), AccumulatorType>();

  const short left_capacity = left_tensor.get_capacity();
  const short right_capacity = right_tensor.get_capacity();
  const short accumulator_capacity = accumulator_tensor.get_capacity();

  short left_col[16], left_row[16];
  short right_col[16], right_row[16];
  short output_col[16], output_row[16];
  bool output_valid[16];

  PRAGMA_UNROLL
  for (short i = 0; i < left_capacity; i++) {
    auto coord = left_tensor.get_multidimensional_index(i);
    left_col[i] = coord[0];
    left_row[i] = coord[1];
  }

  PRAGMA_UNROLL
  for (short i = 0; i < right_capacity; i++) {
    auto coord = right_tensor.get_multidimensional_index(i);
    right_col[i] = coord[0];
    right_row[i] = coord[1];
  }

  PRAGMA_UNROLL
  for (short i = 0; i < accumulator_capacity; i++) {
    auto coord = accumulator_tensor.get_multidimensional_index(i);
    output_col[i] = coord[0];
    output_row[i] = coord[1];
    output_valid[i] = accumulator_tensor.is_valid_element(i);
  }

  AccumulatorType accum_storage[TILES_M * TILES_N][16];
  int all_left_tg_base[TILES_M * TILES_N][16];
  int all_right_tg_base[TILES_M * TILES_N][16];

  PRAGMA_UNROLL
  for (short tile_m = 0; tile_m < TILES_M; tile_m++) {
    PRAGMA_UNROLL
    for (short tile_n = 0; tile_n < TILES_N; tile_n++) {
      const short subtile_index = tile_m * TILES_N + tile_n;
      const short a_m_base = tile_row_offset + tile_m * SUBTILE_ROWS;
      const short b_n_base = tile_col_offset + tile_n * SUBTILE_COLS;

      PRAGMA_UNROLL
      for (short i = 0; i < left_capacity; i++) {
        all_left_tg_base[subtile_index][i] = (a_m_base + left_row[i]) * THREADGROUP_LD + left_col[i];
      }

      PRAGMA_UNROLL
      for (short i = 0; i < right_capacity; i++) {
        all_right_tg_base[subtile_index][i] = (b_n_base + right_row[i]) * THREADGROUP_LD + right_col[i];
      }

      PRAGMA_UNROLL
      for (short i = 0; i < 16; i++) {
        accum_storage[subtile_index][i] = AccumulatorType(0);
      }
    }
  }

  const int full_prefetch_iterations = params->K / PREFETCH_K;
  const int k_remainder = params->K - full_prefetch_iterations * PREFETCH_K;

  const short actual_bm = align_m ? BLOCK_M : short(min(int(BLOCK_M), params->M - block_row_start));
  const short actual_bn = align_n ? BLOCK_N : short(min(int(BLOCK_N), params->N - block_col_start));

  // --- Main K loop ---
  for (int outer_k = 0; outer_k < full_prefetch_iterations; outer_k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (align_m && align_n) {
      loader_a.load_unchecked();
      loader_b.load_unchecked();
    } else {
      loader_a.load_checked(short2(PREFETCH_K, actual_bm));
      loader_b.load_checked(short2(PREFETCH_K, actual_bn));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    PRAGMA_UNROLL
    for (short tile_m = 0; tile_m < TILES_M; tile_m++) {
      PRAGMA_UNROLL
      for (short tile_n = 0; tile_n < TILES_N; tile_n++) {
        const short subtile_index = tile_m * TILES_N + tile_n;

        const short m_limit = is_unaligned_m ? short(max(0, int(simdgroup_limit_m) - tile_m * SUBTILE_ROWS)) : SUBTILE_ROWS;
        const short n_limit = is_unaligned_n ? short(max(0, int(simdgroup_limit_n) - tile_n * SUBTILE_COLS)) : SUBTILE_COLS;
        if (m_limit <= 0 || n_limit <= 0) continue;

        PRAGMA_UNROLL
        for (short i = 0; i < accumulator_capacity; i++)
          accumulator_tensor[i] = accum_storage[subtile_index][i];

        PRAGMA_UNROLL
        for (short k_step = 0; k_step < INNER_K_STEPS; k_step++) {
          const short k_offset = k_step * MATMUL_K_STEP;

          PRAGMA_UNROLL
          for (short i = 0; i < left_capacity; i++)
            left_tensor[i] = left_shared[all_left_tg_base[subtile_index][i] + k_offset];

          PRAGMA_UNROLL
          for (short i = 0; i < right_capacity; i++)
            right_tensor[i] = right_shared[all_right_tg_base[subtile_index][i] + k_offset];

          matmul_operation.run(left_tensor, right_tensor, accumulator_tensor);
        }

        PRAGMA_UNROLL
        for (short i = 0; i < accumulator_capacity; i++)
          accum_storage[subtile_index][i] = accumulator_tensor[i];
      }
    }

    loader_a.next();
    loader_b.next();
  }

  // --- K remainder ---
  if (k_remainder > 0) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_a.load_checked(short2(k_remainder, actual_bm));
    loader_b.load_checked(short2(k_remainder, actual_bn));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short remainder_steps = short((k_remainder + MATMUL_K_STEP - 1) / MATMUL_K_STEP);

    PRAGMA_UNROLL
    for (short tile_m = 0; tile_m < TILES_M; tile_m++) {
      PRAGMA_UNROLL
      for (short tile_n = 0; tile_n < TILES_N; tile_n++) {
        const short subtile_index = tile_m * TILES_N + tile_n;

        const short m_limit = is_unaligned_m ? short(max(0, int(simdgroup_limit_m) - tile_m * SUBTILE_ROWS)) : SUBTILE_ROWS;
        const short n_limit = is_unaligned_n ? short(max(0, int(simdgroup_limit_n) - tile_n * SUBTILE_COLS)) : SUBTILE_COLS;
        if (m_limit <= 0 || n_limit <= 0) continue;

        PRAGMA_UNROLL
        for (short i = 0; i < accumulator_capacity; i++)
          accumulator_tensor[i] = accum_storage[subtile_index][i];

        for (short k_step = 0; k_step < remainder_steps; k_step++) {
          const short k_offset = k_step * MATMUL_K_STEP;

          PRAGMA_UNROLL
          for (short i = 0; i < left_capacity; i++)
            left_tensor[i] = left_shared[all_left_tg_base[subtile_index][i] + k_offset];

          PRAGMA_UNROLL
          for (short i = 0; i < right_capacity; i++)
            right_tensor[i] = right_shared[all_right_tg_base[subtile_index][i] + k_offset];

          matmul_operation.run(left_tensor, right_tensor, accumulator_tensor);
        }

        PRAGMA_UNROLL
        for (short i = 0; i < accumulator_capacity; i++)
          accum_storage[subtile_index][i] = accumulator_tensor[i];
      }
    }
  }

  // --- Store results ---
  PRAGMA_UNROLL
  for (short tile_m = 0; tile_m < TILES_M; tile_m++) {
    PRAGMA_UNROLL
    for (short tile_n = 0; tile_n < TILES_N; tile_n++) {
      const short row_offset = tile_m * SUBTILE_ROWS;
      const short col_offset = tile_n * SUBTILE_COLS;

      const short m_limit = is_unaligned_m ? short(max(0, int(simdgroup_limit_m) - row_offset)) : SUBTILE_ROWS;
      const short n_limit = is_unaligned_n ? short(max(0, int(simdgroup_limit_n) - col_offset)) : SUBTILE_COLS;
      if (m_limit <= 0 || n_limit <= 0) continue;

      device T* output_subtile_ptr = output_ptr + row_offset * params->leading_dim_d + col_offset;

      PRAGMA_UNROLL
      for (short i = 0; i < accumulator_capacity; i++)
        accumulator_tensor[i] = accum_storage[tile_m * TILES_N + tile_n][i];

      const bool subtile_aligned_m = !is_unaligned_m || (m_limit == SUBTILE_ROWS);
      const bool subtile_aligned_n = !is_unaligned_n || (n_limit == SUBTILE_COLS);

      PRAGMA_UNROLL
      for (short i = 0; i < accumulator_capacity; i++) {
        if (subtile_aligned_m && subtile_aligned_n) {
          if (output_valid[i])
            output_subtile_ptr[output_row[i] * params->leading_dim_d + output_col[i]] = T(accumulator_tensor[i]);
        } else {
          if (output_valid[i] && output_row[i] < m_limit && output_col[i] < n_limit)
            output_subtile_ptr[output_row[i] * params->leading_dim_d + output_col[i]] = T(accumulator_tensor[i]);
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// DSL kernel entry point
///////////////////////////////////////////////////////////////////////////////

template <typename T>
VARIANTS(T, half, bfloat)
KERNEL(MatmulGemmMpp)(
    const device T* left_matrix,
    const device T* right_matrix,
    device T* output_matrix,
    const constant uzu::matmul::GEMMParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant uint& group_count_z,
    threadgroup T left_shared[THREADGROUP_TILE_SIZE],
    threadgroup T right_shared[THREADGROUP_TILE_SIZE],
    const uint block_rows SPECIALIZE,
    const uint block_cols SPECIALIZE,
    const uint block_depth SPECIALIZE,
    const uint warps_per_row SPECIALIZE,
    const uint warps_per_col SPECIALIZE,
    const bool align_m SPECIALIZE,
    const bool align_n SPECIALIZE,
    const bool align_k SPECIALIZE,
    const bool use_native_fragment_layout SPECIALIZE,
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint group_z GROUPS(group_count_z),
    const uint thread_x THREADS(32),
    const uint thread_y THREADS(4),
    const uint thread_z THREADS(4),
    const Simd simd
) {
  if (simd.group_idx >= warps_per_row * warps_per_col) {
    return;
  }

  if (block_rows == 128 && block_cols == 128 && block_depth == 512 &&
      warps_per_row == 4 && warps_per_col == 4) {
    gemm_mpp_impl<T, 128, 128, 512, 4, 4>(
        left_matrix, right_matrix, output_matrix, params,
        align_m, align_n, align_k,
        left_shared, right_shared,
        simd.group_idx, simd.lane_idx,
        uint3(group_x, group_y, group_z));
  } else {
    gemm_mpp_impl<T, 64, 64, 256, 2, 2>(
        left_matrix, right_matrix, output_matrix, params,
        align_m, align_n, align_k,
        left_shared, right_shared,
        simd.group_idx, simd.lane_idx,
        uint3(group_x, group_y, group_z));
  }
}

// clang-format on
