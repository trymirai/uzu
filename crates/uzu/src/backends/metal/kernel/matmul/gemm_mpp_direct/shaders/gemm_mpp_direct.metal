// clang-format off
#include "../../../common/dsl.h"
#include "../../../common/thread_context.h"

#include "mxu_gemm_loop.h"

using namespace uzu::matmul;

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    typename AccumType = float>
METAL_FUNC void gemm_mpp_direct_impl(
    const device T* A,
    const device T* B,
    device T* D,
    const constant GemmParams* params,
    const bool align_m,
    const bool align_n,
    const bool align_k,
    uint simd_group_id,
    uint2 threadgroup_position
) {
  const int tid_y = ((threadgroup_position.y) << params->swizzle_log) +
      ((threadgroup_position.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (threadgroup_position.x) >> params->swizzle_log;

  if (int(params->threadgroups_per_row) <= tid_x ||
      int(params->threadgroups_per_column) <= tid_y) {
    return;
  }

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);

  A += c_row_long * params->leading_dimension_a;
  B += c_col_long * params->leading_dimension_b;
  D += c_row_long * params->leading_dimension_d + c_col_long;

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  const int sgp_sm_int =
      align_m ? int(SM) : min(int(SM), int(params->M) - (c_row + tm));
  const short sgp_sm = short(sgp_sm_int);
  const bool is_unaligned_sm = align_m ? false : (sgp_sm != SM);

  const int sgp_sn_int =
      align_n ? int(SN) : min(int(SN), int(params->N) - (c_col + tn));
  const short sgp_sn = short(sgp_sn_int);
  const bool is_unaligned_sn = align_n ? false : (sgp_sn != SN);

  A += tm * params->leading_dimension_a;
  B += tn;
  D += tm * params->leading_dimension_d + tn;

  const int gemm_k_iterations_aligned = int(params->K) / BK;

  dispatch_bool(align_k, [&](auto kAlignedK) {
    dispatch_bool(align_m || !is_unaligned_sm, [&](auto kAlignedM) {
      dispatch_bool(align_n || !is_unaligned_sn, [&](auto kAlignedN) {
        auto Dtile = gemm_loop<
            T, SM, SN, SK, BK,
            false, true,
            kAlignedM.value,
            kAlignedN.value,
            kAlignedK.value,
            AccumType>(
            A, B,
            int(params->leading_dimension_a),
            int(params->leading_dimension_b),
            int(params->K),
            gemm_k_iterations_aligned,
            sgp_sm, sgp_sn);
        if constexpr (kAlignedM && kAlignedN) {
          Dtile.store(D, int(params->leading_dimension_d));
        } else {
          Dtile.store_safe(D, int(params->leading_dimension_d), short2(sgp_sn, sgp_sm));
        }
      });
    });
  });
}

#define GEMM_MPP_DIRECT_DISPATCH(T, BM_, BN_, BK_, WM_, WN_) \
  if (block_rows == BM_ && block_cols == BN_ && \
      bk == BK_ && \
      simdgroups_per_row == WM_ && simdgroups_per_column == WN_) { \
    gemm_mpp_direct_impl<T, BM_, BN_, BK_, WM_, WN_>( \
        left_matrix, right_matrix, output_matrix, params, \
        align_m, align_n, align_k, \
        thread_context.threadgroup_index, \
        uint2(group_x, group_y)); \
    return; \
  }

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MatmulGemmMppDirect)(
    const device T* left_matrix,
    const device T* right_matrix,
    device T* output_matrix,
    const constant uzu::matmul::GemmParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const uint block_rows SPECIALIZE,
    const uint block_cols SPECIALIZE,
    const uint simdgroups_per_row SPECIALIZE,
    const uint simdgroups_per_column SPECIALIZE,
    const bool align_m SPECIALIZE,
    const bool align_n SPECIALIZE,
    const bool align_k SPECIALIZE,
    const uint bk SPECIALIZE,
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint thread_x THREADS(32),
    const uint thread_y THREADS(4),
    const uint thread_z THREADS(4),
    const ThreadContext thread_context
) {
  if (thread_context.threadgroup_index >= simdgroups_per_row * simdgroups_per_column) {
    return;
  }

  GEMM_MPP_DIRECT_DISPATCH(T,  64,  64, 256, 2, 2)
  GEMM_MPP_DIRECT_DISPATCH(T, 128, 128, 256, 4, 4)
}

#undef GEMM_MPP_DIRECT_DISPATCH
// clang-format on
