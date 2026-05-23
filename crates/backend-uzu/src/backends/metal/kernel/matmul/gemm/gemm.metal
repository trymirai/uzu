#include "../../common/dsl.h"
#include "../../common/defines.h"
#include "../../common/thread_context.h"
#include "../generated/gemm.h"

#include "common/mxu_mma_core.h"
#include "common/simdgroup_mma_core.h"

using namespace metal;
using namespace uzu::gemm;

#define GEMM_MAX_THREADGROUP_A 2560
#define GEMM_MAX_THREADGROUP_B 1536

template <
    typename T,
    uint THREADGROUP_BLOCK_M,
    uint THREADGROUP_BLOCK_N,
    uint THREADGROUP_BLOCK_K,
    uint SIMDGROUPS_M,
    uint SIMDGROUPS_N,
    bool TRANSPOSE_B,
    bool USE_MXU>
VARIANTS(T, float, half, bfloat)
VARIANTS(THREADGROUP_BLOCK_M, 32, 64, 128)
VARIANTS(THREADGROUP_BLOCK_N, 32, 64, 128)
VARIANTS(THREADGROUP_BLOCK_K, 16, 32)
VARIANTS(SIMDGROUPS_M, 1, 2, 4)
VARIANTS(SIMDGROUPS_N, 1, 2, 4)
VARIANTS(TRANSPOSE_B, false, true)
VARIANTS(USE_MXU, false, true)
CONSTRAINT(!USE_MXU || THREADGROUP_BLOCK_M % SIMDGROUPS_M == 0)
CONSTRAINT(!USE_MXU || THREADGROUP_BLOCK_N % SIMDGROUPS_N == 0)
CONSTRAINT(!USE_MXU || (THREADGROUP_BLOCK_M / SIMDGROUPS_M) % 16 == 0)
CONSTRAINT(!USE_MXU || (THREADGROUP_BLOCK_N / SIMDGROUPS_N) % 16 == 0)
CONSTRAINT(USE_MXU || max(THREADGROUP_BLOCK_M, THREADGROUP_BLOCK_N) <= 32 * SIMDGROUPS_M * SIMDGROUPS_N)
KERNEL(Gemm)(
    const device T* a,
    const device uint8_t* b_packed,
    device T* d,
    const device T* scales
        OPTIONAL(weight_prologue != GemmWeightPrologueKind::FullPrecision),
    const device T* biases
        OPTIONAL(weight_prologue == GemmWeightPrologueKind::ScaleBiasDequant),
    const device uint8_t* zero_points
        OPTIONAL(weight_prologue == GemmWeightPrologueKind::ScaleZeroPointDequant),
    const constant uzu::matmul::GemmParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const GemmInputPrologueKind input_prologue SPECIALIZE,
    const GemmWeightPrologueKind weight_prologue SPECIALIZE,
    const GemmOutputTransformKind output_transform SPECIALIZE,
    const GemmAlignment alignment SPECIALIZE,
    const uint bits_per_weight SPECIALIZE,
    const uint group_size SPECIALIZE,
    threadgroup T a_shared[GEMM_MAX_THREADGROUP_A],
    threadgroup T b_shared[GEMM_MAX_THREADGROUP_B],
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint thread_x THREADS(METAL_SIMD_SIZE),
    const uint thread_y THREADS(SIMDGROUPS_N),
    const uint thread_z THREADS(SIMDGROUPS_M),
    const ThreadContext thread_context
) {
  (void)scales;
  (void)biases;
  (void)zero_points;
  (void)bits_per_weight;
  (void)group_size;
  (void)group_x;
  (void)group_y;
  (void)thread_x;
  (void)thread_y;
  (void)thread_z;
  (void)input_prologue;
  (void)weight_prologue;
  const device T* b = reinterpret_cast<const device T*>(b_packed);
  if constexpr (USE_MXU) {
    MxuMmaCore<
        T,
        THREADGROUP_BLOCK_M,
        THREADGROUP_BLOCK_N,
        SIMDGROUPS_M,
        SIMDGROUPS_N,
        TRANSPOSE_B>::
        run(a, b, d, params, alignment, output_transform, thread_context);
  } else {
    SimdgroupMmaCore<
        T,
        THREADGROUP_BLOCK_M,
        THREADGROUP_BLOCK_N,
        THREADGROUP_BLOCK_K,
        SIMDGROUPS_M,
        SIMDGROUPS_N,
        TRANSPOSE_B>::
        run(a,
            b,
            d,
            params,
            alignment,
            output_transform,
            a_shared,
            b_shared,
            thread_context);
  }
}
