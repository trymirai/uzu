// clang-format off
#include "../../../common/utils.h"
#include "../../../definitions.metal"
#include "../../common/steel/gemm/params.h"

using namespace metal;

namespace uzu {
namespace matmul {
using GEMMParams = steel::GEMMParams;
} // namespace matmul
} // namespace uzu

///////////////////////////////////////////////////////////////////////////////
// Simple tiled GEMM for mixed types
//
// Each threadgroup computes a TILE_M x TILE_N output tile.
// The K dimension is iterated in chunks of TILE_K through threadgroup memory.
// transpose_a=false, transpose_b=true (matches existing matmul convention).
///////////////////////////////////////////////////////////////////////////////

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

#define THREADS_PER_GROUP 256

template <typename AType, typename BType, typename OutType, typename AccumType>
METAL_FUNC void mixed_types_simple_gemm_impl(
    const device AType* a,
    const device BType* b,
    device OutType* d,
    const constant uzu::matmul::GEMMParams* params,
    threadgroup AType* a_shared,
    threadgroup BType* b_shared,
    uint3 tid,
    uint3 lid,
    uint thread_idx
) {
  const int M = params->M;
  const int N = params->N;
  const int K = params->K;
  const int lda = params->lda;
  const int ldb = params->ldb;
  const int ldd = params->ldd;

  a += params->batch_stride_a * tid.z;
  b += params->batch_stride_b * tid.z;
  d += params->batch_stride_d * tid.z;

  const int row_start = tid.y * TILE_M;
  const int col_start = tid.x * TILE_N;

  const int local_row = lid.y;
  const int local_col = lid.x;

  AccumType acc = AccumType(0);

  for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cooperative load A tile [TILE_M x TILE_K] into threadgroup memory
    // A is row-major: a[row * lda + col], transpose_a=false
    for (int idx = thread_idx; idx < TILE_M * TILE_K; idx += THREADS_PER_GROUP) {
      const int lr = idx / TILE_K;
      const int lc = idx % TILE_K;
      const int gr = row_start + lr;
      const int gc = k_tile + lc;
      a_shared[lr * TILE_K + lc] = (gr < M && gc < K) ? a[gr * lda + gc] : AType(0);
    }

    // Cooperative load B tile [TILE_N x TILE_K] into threadgroup memory
    // B is row-major with transpose_b=true: b[col * ldb + k]
    for (int idx = thread_idx; idx < TILE_N * TILE_K; idx += THREADS_PER_GROUP) {
      const int lr = idx / TILE_K;
      const int lc = idx % TILE_K;
      const int gr = col_start + lr;
      const int gc = k_tile + lc;
      b_shared[lr * TILE_K + lc] = (gr < N && gc < K) ? b[gr * ldb + gc] : BType(0);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread computes one element of the output tile
    if (local_row < TILE_M && local_col < TILE_N) {
      const int k_end = min(TILE_K, K - k_tile);
      for (int kk = 0; kk < k_end; kk++) {
        acc += AccumType(a_shared[local_row * TILE_K + kk]) *
               AccumType(b_shared[local_col * TILE_K + kk]);
      }
    }
  }

  // Write output
  const int out_row = row_start + local_row;
  const int out_col = col_start + local_col;
  if (out_row < M && out_col < N && local_row < TILE_M && local_col < TILE_N) {
    d[out_row * ldd + out_col] = OutType(acc);
  }
}

///////////////////////////////////////////////////////////////////////////////
// DSL kernel entry points
///////////////////////////////////////////////////////////////////////////////

// i8 * i8 -> i32 (accumulate in int)
KERNEL(MixedTypesSimpleGemmI8I8I32)(
    const device int8_t* a,
    const device int8_t* b,
    device int* d,
    const constant uzu::matmul::GEMMParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant uint& group_count_z,
    threadgroup int8_t a_shared[TILE_M * TILE_K],
    threadgroup int8_t b_shared[TILE_N * TILE_K],
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint group_z GROUPS(group_count_z),
    const uint thread_x THREADS(32),
    const uint thread_y THREADS(8),
    const uint thread_z THREADS(1)
) {
  const uint thread_idx = thread_y * 32 + thread_x;
  mixed_types_simple_gemm_impl<int8_t, int8_t, int, int>(
      a, b, d, params,
      a_shared, b_shared,
      uint3(group_x, group_y, group_z),
      uint3(thread_x, thread_y, 0),
      thread_idx
  );
}

// i8 * bf16 -> bf16 (accumulate in float)
KERNEL(MixedTypesSimpleGemmI8Bf16Bf16)(
    const device int8_t* a,
    const device bfloat16_t* b,
    device bfloat16_t* d,
    const constant uzu::matmul::GEMMParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant uint& group_count_z,
    threadgroup int8_t a_shared[TILE_M * TILE_K],
    threadgroup bfloat16_t b_shared[TILE_N * TILE_K],
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint group_z GROUPS(group_count_z),
    const uint thread_x THREADS(32),
    const uint thread_y THREADS(8),
    const uint thread_z THREADS(1)
) {
  const uint thread_idx = thread_y * 32 + thread_x;
  mixed_types_simple_gemm_impl<int8_t, bfloat16_t, bfloat16_t, float>(
      a, b, d, params,
      a_shared, b_shared,
      uint3(group_x, group_y, group_z),
      uint3(thread_x, thread_y, 0),
      thread_idx
  );
}

// i8 * f16 -> f16 (accumulate in float)
KERNEL(MixedTypesSimpleGemmI8F16F16)(
    const device int8_t* a,
    const device half* b,
    device half* d,
    const constant uzu::matmul::GEMMParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant uint& group_count_z,
    threadgroup int8_t a_shared[TILE_M * TILE_K],
    threadgroup half b_shared[TILE_N * TILE_K],
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint group_z GROUPS(group_count_z),
    const uint thread_x THREADS(32),
    const uint thread_y THREADS(8),
    const uint thread_z THREADS(1)
) {
  const uint thread_idx = thread_y * 32 + thread_x;
  mixed_types_simple_gemm_impl<int8_t, half, half, float>(
      a, b, d, params,
      a_shared, b_shared,
      uint3(group_x, group_y, group_z),
      uint3(thread_x, thread_y, 0),
      thread_idx
  );
}

// i8 * f32 -> f32 (accumulate in float)
KERNEL(MixedTypesSimpleGemmI8F32F32)(
    const device int8_t* a,
    const device float* b,
    device float* d,
    const constant uzu::matmul::GEMMParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant uint& group_count_z,
    threadgroup int8_t a_shared[TILE_M * TILE_K],
    threadgroup float b_shared[TILE_N * TILE_K],
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint group_z GROUPS(group_count_z),
    const uint thread_x THREADS(32),
    const uint thread_y THREADS(8),
    const uint thread_z THREADS(1)
) {
  const uint thread_idx = thread_y * 32 + thread_x;
  mixed_types_simple_gemm_impl<int8_t, float, float, float>(
      a, b, d, params,
      a_shared, b_shared,
      uint3(group_x, group_y, group_z),
      uint3(thread_x, thread_y, 0),
      thread_idx
  );
}

// clang-format on
