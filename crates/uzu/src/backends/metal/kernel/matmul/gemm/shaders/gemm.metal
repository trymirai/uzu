// clang-format off
#include "../../../common/utils.h"
#include "../../../definitions.metal"

#include "../../common/gemm.h"

using namespace uzu::matmul;

template <typename T, int BM, int BN, int BK, int WM, int WN, bool MN_aligned, bool K_aligned>
METAL_FUNC void gemm_dispatch(
    const device T* a,
    const device T* b,
    device T* d,
    const constant GEMMParams* params,
    threadgroup T* a_shared,
    threadgroup T* b_shared,
    uint simd_lane_id,
    uint simd_group_id,
    uint3 threadgroup_position,
    uint3 thread_position
) {
  using gemm_kernel = GEMMKernel<T, T, BM, BN, BK, WM, WN, false, true, MN_aligned, K_aligned, float>;
  gemm_kernel::run(a, b, d, params, a_shared, b_shared, simd_lane_id, simd_group_id, threadgroup_position, thread_position);
}

#define GEMM_TILE(T, BM, BN, BK, WM, WN) \
  if (block_rows == BM && block_cols == BN && block_depth == BK && \
      warps_per_row == WM && warps_per_col == WN) { \
    if (align_m && align_n) { \
      if (align_k) \
        gemm_dispatch<T, BM, BN, BK, WM, WN, true, true>(a, b, d, params, a_shared, b_shared, simd.lane_idx, simd.group_idx, uint3(group_x, group_y, group_z), uint3(thread_x, thread_y, thread_z)); \
      else \
        gemm_dispatch<T, BM, BN, BK, WM, WN, true, false>(a, b, d, params, a_shared, b_shared, simd.lane_idx, simd.group_idx, uint3(group_x, group_y, group_z), uint3(thread_x, thread_y, thread_z)); \
    } else { \
      if (align_k) \
        gemm_dispatch<T, BM, BN, BK, WM, WN, false, true>(a, b, d, params, a_shared, b_shared, simd.lane_idx, simd.group_idx, uint3(group_x, group_y, group_z), uint3(thread_x, thread_y, thread_z)); \
      else \
        gemm_dispatch<T, BM, BN, BK, WM, WN, false, false>(a, b, d, params, a_shared, b_shared, simd.lane_idx, simd.group_idx, uint3(group_x, group_y, group_z), uint3(thread_x, thread_y, thread_z)); \
    } \
    return; \
  }

#define GEMM_MAX_TGP_A 2560
#define GEMM_MAX_TGP_B 1536

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MatmulGemm)(
    const device T* a,
    const device T* b,
    device T* d,
    const constant uzu::matmul::GEMMParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant uint& group_count_z,
    threadgroup T a_shared[GEMM_MAX_TGP_A],
    threadgroup T b_shared[GEMM_MAX_TGP_B],
    const uint block_rows SPECIALIZE,
    const uint block_cols SPECIALIZE,
    const uint block_depth SPECIALIZE,
    const uint warps_per_row SPECIALIZE,
    const uint warps_per_col SPECIALIZE,
    const bool align_m SPECIALIZE,
    const bool align_n SPECIALIZE,
    const bool align_k SPECIALIZE,
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint group_z GROUPS(group_count_z),
    const uint thread_x THREADS(32),
    const uint thread_y THREADS(2),
    const uint thread_z THREADS(2),
    const Simd simd
) {
  GEMM_TILE(T, 64, 64, 16, 2, 2)
  GEMM_TILE(T, 64, 64, 16, 1, 2)
  GEMM_TILE(T, 64, 32, 32, 2, 2)
  GEMM_TILE(T, 32, 64, 16, 1, 2)
  GEMM_TILE(T, 32, 64, 16, 2, 2)
  GEMM_TILE(T, 32, 32, 16, 2, 2)
  GEMM_TILE(T, 64, 32,  8, 4, 1)
}

#undef GEMM_TILE
// clang-format on
