// clang-format off
#include "../../../common/utils.h"
#include "../../../definitions.metal"

#include "../../common/mpp_gemm_utilities.h"
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
    short BM,
    short BN,
    short BK,
    short WM,
    short WN>
METAL_FUNC void gemm_mpp_impl(
    const device T* a,
    const device T* b,
    device T* d,
    const constant GEMMParams* params,
    const bool align_m,
    const bool align_n,
    const bool align_k,
    threadgroup T* a_shared,
    threadgroup T* b_shared,
    uint simd_group_id,
    uint simd_lane_id,
    uint3 tid
) {
  using AccumType = float;

  constexpr short SUBTILE_ROWS = 16;
  constexpr short SUBTILE_COLS = 32;
  constexpr short MATMUL_K_STEP = 16;
  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short TM = SM / SUBTILE_ROWS;
  constexpr short TN = SN / SUBTILE_COLS;
  constexpr short TGP_SIZE = WM * WN * 32;

  constexpr short PREFETCH_K = short(PREFETCH_K_SIZE);
  constexpr short TG_LD = PREFETCH_K + short(16 / sizeof(T));
  constexpr short INNER_K_STEPS = PREFETCH_K / MATMUL_K_STEP;

  const int tid_y = ((tid.y) << params->swizzle_log) +
                    ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;

  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  a += params->batch_stride_a * tid.z;
  b += params->batch_stride_b * tid.z;
  d += params->batch_stride_d * tid.z;

  threadgroup_barrier(mem_flags::mem_none);

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);

  const device T* a_block = a + c_row_long * params->lda;
  const device T* b_block = b + c_col_long * params->ldb;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  device T* d_out = d + c_row_long * params->ldd + c_col_long + tm * params->ldd + tn;

  const int sgp_sm_int =
      align_m ? int(SM) : min(int(SM), params->M - (c_row + tm));
  const short sgp_sm = short(sgp_sm_int);

  const int sgp_sn_int =
      align_n ? int(SN) : min(int(SN), params->N - (c_col + tn));
  const short sgp_sn = short(sgp_sn_int);

  const bool is_unaligned_sm = align_m ? false : (sgp_sm != SM);
  const bool is_unaligned_sn = align_n ? false : (sgp_sn != SN);

  // --- Fallback: direct device reads when staging gives no benefit ---
  if constexpr (PREFETCH_K <= MATMUL_K_STEP) {
    const device T* a_sg = a_block + tm * params->lda;
    const device T* b_sg = b_block + tn * params->ldb;

    UZU_PRAGMA_UNROLL
    for (short mm = 0; mm < TM; mm++) {
      UZU_PRAGMA_UNROLL
      for (short nn = 0; nn < TN; nn++) {
        const short m_off = mm * SUBTILE_ROWS;
        const short n_off = nn * SUBTILE_COLS;

        const short m_limit = is_unaligned_sm ? short(max(0, int(sgp_sm) - m_off)) : SUBTILE_ROWS;
        const short n_limit = is_unaligned_sn ? short(max(0, int(sgp_sn) - n_off)) : SUBTILE_COLS;
        if (m_limit <= 0 || n_limit <= 0) continue;

        const device T* a_sub = a_sg + m_off * params->lda;
        const device T* b_sub = b_sg + n_off * params->ldb;
        device T* d_sub = d_out + m_off * params->ldd + n_off;

        const bool subtile_aligned_m = !is_unaligned_sm || (m_limit == SUBTILE_ROWS);
        const bool subtile_aligned_n = !is_unaligned_sn || (n_limit == SUBTILE_COLS);

        if (subtile_aligned_m && subtile_aligned_n && align_k) {
          cooperative_tensor_gemm<SUBTILE_ROWS, SUBTILE_COLS, MATMUL_K_STEP, AccumType, T, T, T,
                                 false, true, true, true, true>(
              a_sub, params->lda, b_sub, params->ldb, d_sub, params->ldd,
              params->K, m_limit, n_limit);
        } else if (subtile_aligned_m && subtile_aligned_n) {
          cooperative_tensor_gemm<SUBTILE_ROWS, SUBTILE_COLS, MATMUL_K_STEP, AccumType, T, T, T,
                                 false, true, true, true, false>(
              a_sub, params->lda, b_sub, params->ldb, d_sub, params->ldd,
              params->K, m_limit, n_limit);
        } else {
          cooperative_tensor_gemm<SUBTILE_ROWS, SUBTILE_COLS, MATMUL_K_STEP, AccumType, T, T, T,
                                 false, true, false, false, false>(
              a_sub, params->lda, b_sub, params->ldb, d_sub, params->ldd,
              params->K, m_limit, n_limit);
        }
      }
    }
    return;
  }

  // --- Staged path: threadgroup memory prefetch ---

  BlockLoader<T, BM, PREFETCH_K, TG_LD, 1, TGP_SIZE> loader_a(
      a_block, params->lda, a_shared, ushort(simd_group_id), ushort(simd_lane_id));
  BlockLoader<T, BN, PREFETCH_K, TG_LD, 1, TGP_SIZE> loader_b(
      b_block, params->ldb, b_shared, ushort(simd_group_id), ushort(simd_lane_id));

  constexpr auto matmul_descriptor = mpp::tensor_ops::matmul2d_descriptor(
      SUBTILE_ROWS, SUBTILE_COLS, MATMUL_K_STEP,
      false, true, false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

  mpp::tensor_ops::matmul2d<matmul_descriptor, metal::execution_simdgroup> matmul_operation;

  auto left_tensor = matmul_operation.template get_left_input_cooperative_tensor<T, T, AccumType>();
  auto right_tensor = matmul_operation.template get_right_input_cooperative_tensor<T, T, AccumType>();
  auto accumulator_tensor = matmul_operation.template get_destination_cooperative_tensor<
      decltype(left_tensor), decltype(right_tensor), AccumType>();

  const short left_capacity = left_tensor.get_capacity();
  const short right_capacity = right_tensor.get_capacity();
  const short accumulator_capacity = accumulator_tensor.get_capacity();

  short left_col[16], left_row[16];
  short right_col[16], right_row[16];
  short output_col[16], output_row[16];
  bool output_valid[16];

  UZU_PRAGMA_UNROLL
  for (short i = 0; i < left_capacity; i++) {
    auto coord = left_tensor.get_multidimensional_index(i);
    left_col[i] = coord[0];
    left_row[i] = coord[1];
  }

  UZU_PRAGMA_UNROLL
  for (short i = 0; i < right_capacity; i++) {
    auto coord = right_tensor.get_multidimensional_index(i);
    right_col[i] = coord[0];
    right_row[i] = coord[1];
  }

  UZU_PRAGMA_UNROLL
  for (short i = 0; i < accumulator_capacity; i++) {
    auto coord = accumulator_tensor.get_multidimensional_index(i);
    output_col[i] = coord[0];
    output_row[i] = coord[1];
    output_valid[i] = accumulator_tensor.is_valid_element(i);
  }

  AccumType accum_storage[TM * TN][16];
  int all_left_tg_base[TM * TN][16];
  int all_right_tg_base[TM * TN][16];

  UZU_PRAGMA_UNROLL
  for (short mm = 0; mm < TM; mm++) {
    UZU_PRAGMA_UNROLL
    for (short nn = 0; nn < TN; nn++) {
      const short subtile_index = mm * TN + nn;
      const short a_m_base = tm + mm * SUBTILE_ROWS;
      const short b_n_base = tn + nn * SUBTILE_COLS;

      UZU_PRAGMA_UNROLL
      for (short i = 0; i < left_capacity; i++)
        all_left_tg_base[subtile_index][i] = (a_m_base + left_row[i]) * TG_LD + left_col[i];

      UZU_PRAGMA_UNROLL
      for (short i = 0; i < right_capacity; i++)
        all_right_tg_base[subtile_index][i] = (b_n_base + right_row[i]) * TG_LD + right_col[i];

      UZU_PRAGMA_UNROLL
      for (short i = 0; i < 16; i++)
        accum_storage[subtile_index][i] = AccumType(0);
    }
  }

  const int full_prefetch_iterations = params->K / PREFETCH_K;
  const int k_remainder = params->K - full_prefetch_iterations * PREFETCH_K;

  const short actual_bm = align_m ? BM : short(min(int(BM), params->M - c_row));
  const short actual_bn = align_n ? BN : short(min(int(BN), params->N - c_col));

  // --- Main K loop ---
  for (int outer_k = 0; outer_k < full_prefetch_iterations; outer_k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (align_m && align_n) {
      loader_a.load_unsafe();
      loader_b.load_unsafe();
    } else {
      loader_a.load_safe(short2(PREFETCH_K, actual_bm));
      loader_b.load_safe(short2(PREFETCH_K, actual_bn));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    UZU_PRAGMA_UNROLL
    for (short mm = 0; mm < TM; mm++) {
      UZU_PRAGMA_UNROLL
      for (short nn = 0; nn < TN; nn++) {
        const short subtile_index = mm * TN + nn;

        const short m_limit = is_unaligned_sm ? short(max(0, int(sgp_sm) - mm * SUBTILE_ROWS)) : SUBTILE_ROWS;
        const short n_limit = is_unaligned_sn ? short(max(0, int(sgp_sn) - nn * SUBTILE_COLS)) : SUBTILE_COLS;
        if (m_limit <= 0 || n_limit <= 0) continue;

        UZU_PRAGMA_UNROLL
        for (short i = 0; i < accumulator_capacity; i++)
          accumulator_tensor[i] = accum_storage[subtile_index][i];

        UZU_PRAGMA_UNROLL
        for (short k_step = 0; k_step < INNER_K_STEPS; k_step++) {
          const short k_offset = k_step * MATMUL_K_STEP;

          UZU_PRAGMA_UNROLL
          for (short i = 0; i < left_capacity; i++)
            left_tensor[i] = a_shared[all_left_tg_base[subtile_index][i] + k_offset];

          UZU_PRAGMA_UNROLL
          for (short i = 0; i < right_capacity; i++)
            right_tensor[i] = b_shared[all_right_tg_base[subtile_index][i] + k_offset];

          matmul_operation.run(left_tensor, right_tensor, accumulator_tensor);
        }

        UZU_PRAGMA_UNROLL
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
    loader_a.load_safe(short2(k_remainder, actual_bm));
    loader_b.load_safe(short2(k_remainder, actual_bn));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short remainder_steps = short((k_remainder + MATMUL_K_STEP - 1) / MATMUL_K_STEP);

    UZU_PRAGMA_UNROLL
    for (short mm = 0; mm < TM; mm++) {
      UZU_PRAGMA_UNROLL
      for (short nn = 0; nn < TN; nn++) {
        const short subtile_index = mm * TN + nn;

        const short m_limit = is_unaligned_sm ? short(max(0, int(sgp_sm) - mm * SUBTILE_ROWS)) : SUBTILE_ROWS;
        const short n_limit = is_unaligned_sn ? short(max(0, int(sgp_sn) - nn * SUBTILE_COLS)) : SUBTILE_COLS;
        if (m_limit <= 0 || n_limit <= 0) continue;

        UZU_PRAGMA_UNROLL
        for (short i = 0; i < accumulator_capacity; i++)
          accumulator_tensor[i] = accum_storage[subtile_index][i];

        for (short k_step = 0; k_step < remainder_steps; k_step++) {
          const short k_offset = k_step * MATMUL_K_STEP;

          UZU_PRAGMA_UNROLL
          for (short i = 0; i < left_capacity; i++)
            left_tensor[i] = a_shared[all_left_tg_base[subtile_index][i] + k_offset];

          UZU_PRAGMA_UNROLL
          for (short i = 0; i < right_capacity; i++)
            right_tensor[i] = b_shared[all_right_tg_base[subtile_index][i] + k_offset];

          matmul_operation.run(left_tensor, right_tensor, accumulator_tensor);
        }

        UZU_PRAGMA_UNROLL
        for (short i = 0; i < accumulator_capacity; i++)
          accum_storage[subtile_index][i] = accumulator_tensor[i];
      }
    }
  }

  // --- Store results ---
  UZU_PRAGMA_UNROLL
  for (short mm = 0; mm < TM; mm++) {
    UZU_PRAGMA_UNROLL
    for (short nn = 0; nn < TN; nn++) {
      const short m_off = mm * SUBTILE_ROWS;
      const short n_off = nn * SUBTILE_COLS;

      const short m_limit = is_unaligned_sm ? short(max(0, int(sgp_sm) - m_off)) : SUBTILE_ROWS;
      const short n_limit = is_unaligned_sn ? short(max(0, int(sgp_sn) - n_off)) : SUBTILE_COLS;
      if (m_limit <= 0 || n_limit <= 0) continue;

      device T* d_sub = d_out + m_off * params->ldd + n_off;

      UZU_PRAGMA_UNROLL
      for (short i = 0; i < accumulator_capacity; i++)
        accumulator_tensor[i] = accum_storage[mm * TN + nn][i];

      const bool subtile_aligned_m = !is_unaligned_sm || (m_limit == SUBTILE_ROWS);
      const bool subtile_aligned_n = !is_unaligned_sn || (n_limit == SUBTILE_COLS);

      UZU_PRAGMA_UNROLL
      for (short i = 0; i < accumulator_capacity; i++) {
        if (subtile_aligned_m && subtile_aligned_n) {
          if (output_valid[i])
            d_sub[output_row[i] * params->ldd + output_col[i]] = T(accumulator_tensor[i]);
        } else {
          if (output_valid[i] && output_row[i] < m_limit && output_col[i] < n_limit)
            d_sub[output_row[i] * params->ldd + output_col[i]] = T(accumulator_tensor[i]);
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
    const device T* a,
    const device T* b,
    device T* d,
    const constant uzu::matmul::GEMMParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant uint& group_count_z,
    threadgroup T a_shared[THREADGROUP_TILE_SIZE],
    threadgroup T b_shared[THREADGROUP_TILE_SIZE],
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
        a, b, d, params,
        align_m, align_n, align_k,
        a_shared, b_shared,
        simd.group_idx, simd.lane_idx,
        uint3(group_x, group_y, group_z));
  } else {
    gemm_mpp_impl<T, 64, 64, 256, 2, 2>(
        a, b, d, params,
        align_m, align_n, align_k,
        a_shared, b_shared,
        simd.group_idx, simd.lane_idx,
        uint3(group_x, group_y, group_z));
  }
}

// clang-format on
