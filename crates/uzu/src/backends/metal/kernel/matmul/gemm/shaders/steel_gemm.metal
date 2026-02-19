

// clang-format off
#include "../../../common/utils.h"
#include "../../../definitions.metal"

#include "../../common/steel/gemm/gemm.h"
#include "steel_gemm.h"

namespace uzu {
namespace matmul {
using GEMMParams = steel::GEMMParams;
} // namespace matmul
} // namespace uzu

template <typename T, int BM, int BN, int BK, int WM, int WN>
METAL_FUNC void run_matmul_gemm_shape(
    const device T* a,
    const device T* b,
    device T* d,
    const constant uzu::matmul::GEMMParams* params,
    const bool align_m,
    const bool align_n,
    const bool align_k,
    threadgroup T* a_shared,
    threadgroup T* b_shared,
    const uint3 threadgroup_position,
    const uint3 thread_position,
    const Simd simd
) {
  gemm_impl<T, BM, BN, BK, WM, WN, false, true, float>(
      a,
      b,
      (const device T*)nullptr,
      d,
      params,
      (const constant GEMMAddMMParams*)nullptr,
      (const constant int*)nullptr,
      (const constant int64_t*)nullptr,
      false,
      false,
      false,
      align_m,
      align_n,
      align_k,
      a_shared,
      b_shared,
      simd.lane_idx,
      simd.group_idx,
      threadgroup_position,
      thread_position
  );
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MatmulGemmTile64x64x16Warp2x2)(
    const device T* a,
    const device T* b,
    device T* d,
    const constant uzu::matmul::GEMMParams* params,
    const bool align_m SPECIALIZE,
    const bool align_n SPECIALIZE,
    const bool align_k SPECIALIZE,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant uint& group_count_z,
    threadgroup T a_shared[GEMMKernel<T, T, 64, 64, 16, 2, 2, false, true, true, true, float>::tgp_mem_size_a],
    threadgroup T b_shared[GEMMKernel<T, T, 64, 64, 16, 2, 2, false, true, true, true, float>::tgp_mem_size_b],
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint group_z GROUPS(group_count_z),
    const uint thread_x THREADS(32),
    const uint thread_y THREADS(2),
    const uint thread_z THREADS(2),
    const Simd simd
) {
  run_matmul_gemm_shape<T, 64, 64, 16, 2, 2>(
      a,
      b,
      d,
      params,
      align_m,
      align_n,
      align_k,
      a_shared,
      b_shared,
      uint3(group_x, group_y, group_z),
      uint3(thread_x, thread_y, thread_z),
      simd
  );
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MatmulGemmTile64x64x16Warp1x2)(
    const device T* a,
    const device T* b,
    device T* d,
    const constant uzu::matmul::GEMMParams* params,
    const bool align_m SPECIALIZE,
    const bool align_n SPECIALIZE,
    const bool align_k SPECIALIZE,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant uint& group_count_z,
    threadgroup T a_shared[GEMMKernel<T, T, 64, 64, 16, 1, 2, false, true, true, true, float>::tgp_mem_size_a],
    threadgroup T b_shared[GEMMKernel<T, T, 64, 64, 16, 1, 2, false, true, true, true, float>::tgp_mem_size_b],
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint group_z GROUPS(group_count_z),
    const uint thread_x THREADS(32),
    const uint thread_y THREADS(2),
    const uint thread_z THREADS(1),
    const Simd simd
) {
  run_matmul_gemm_shape<T, 64, 64, 16, 1, 2>(
      a,
      b,
      d,
      params,
      align_m,
      align_n,
      align_k,
      a_shared,
      b_shared,
      uint3(group_x, group_y, group_z),
      uint3(thread_x, thread_y, thread_z),
      simd
  );
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MatmulGemmTile64x32x32Warp2x2)(
    const device T* a,
    const device T* b,
    device T* d,
    const constant uzu::matmul::GEMMParams* params,
    const bool align_m SPECIALIZE,
    const bool align_n SPECIALIZE,
    const bool align_k SPECIALIZE,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant uint& group_count_z,
    threadgroup T a_shared[GEMMKernel<T, T, 64, 32, 32, 2, 2, false, true, true, true, float>::tgp_mem_size_a],
    threadgroup T b_shared[GEMMKernel<T, T, 64, 32, 32, 2, 2, false, true, true, true, float>::tgp_mem_size_b],
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint group_z GROUPS(group_count_z),
    const uint thread_x THREADS(32),
    const uint thread_y THREADS(2),
    const uint thread_z THREADS(2),
    const Simd simd
) {
  run_matmul_gemm_shape<T, 64, 32, 32, 2, 2>(
      a,
      b,
      d,
      params,
      align_m,
      align_n,
      align_k,
      a_shared,
      b_shared,
      uint3(group_x, group_y, group_z),
      uint3(thread_x, thread_y, thread_z),
      simd
  );
}

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MatmulGemmTile32x64x16Warp1x2)(
    const device T* a,
    const device T* b,
    device T* d,
    const constant uzu::matmul::GEMMParams* params,
    const bool align_m SPECIALIZE,
    const bool align_n SPECIALIZE,
    const bool align_k SPECIALIZE,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant uint& group_count_z,
    threadgroup T a_shared[GEMMKernel<T, T, 32, 64, 16, 1, 2, false, true, true, true, float>::tgp_mem_size_a],
    threadgroup T b_shared[GEMMKernel<T, T, 32, 64, 16, 1, 2, false, true, true, true, float>::tgp_mem_size_b],
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint group_z GROUPS(group_count_z),
    const uint thread_x THREADS(32),
    const uint thread_y THREADS(2),
    const uint thread_z THREADS(1),
    const Simd simd
) {
  run_matmul_gemm_shape<T, 32, 64, 16, 1, 2>(
      a,
      b,
      d,
      params,
      align_m,
      align_n,
      align_k,
      a_shared,
      b_shared,
      uint3(group_x, group_y, group_z),
      uint3(thread_x, thread_y, thread_z),
      simd
  );
}

// clang-format on
