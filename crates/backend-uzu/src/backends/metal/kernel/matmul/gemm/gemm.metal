#include "../../common/dsl.h"
#include "../../common/defines.h"
#include "../../common/thread_context.h"
#include "../generated/gemm.h"

#include "common/gemm_tiling.h"
#include "common/mxu_mma_core.h"
#include "common/simdgroup_mma_core.h"

using namespace metal;
using namespace uzu::gemm;

#define GEMM_TGA_ELEMENTS                                                      \
  (USE_MXU ? 1                                                                 \
           : (gemm_tiling_block_m(GEMM_TILING) *                               \
              (gemm_tiling_block_k(GEMM_TILING) + 16 / int(sizeof(T)))))
#define GEMM_TGB_ELEMENTS                                                      \
  (USE_MXU ? 1                                                                 \
           : (gemm_tiling_block_n(GEMM_TILING) *                               \
              (gemm_tiling_block_k(GEMM_TILING) + 16 / int(sizeof(T)))))

template <
    typename T,
    GemmTiling GEMM_TILING,
    bool TRANSPOSE_B,
    bool USE_MXU,
    GemmWeightPrologueKind WEIGHT_PROLOGUE,
    uint BITS,
    uint GROUP_SIZE>
VARIANTS(T, half, bfloat)
VARIANTS(
    GEMM_TILING,
    GemmTiling::T8x32x32_1x1,
    GemmTiling::T64x32x32_2x2,
    GemmTiling::T64x64x16_2x2,
    GemmTiling::T64x64x32_2x2,
    GemmTiling::T64x64x64_2x2,
    GemmTiling::T32x32x32_2x2,
    GemmTiling::T32x64x32_2x2,
    GemmTiling::T64x32x32_4x1,
    GemmTiling::T128x128x32_4x4)
VARIANTS(TRANSPOSE_B, false, true)
VARIANTS(USE_MXU, false, true)
VARIANTS(
    WEIGHT_PROLOGUE,
    GemmWeightPrologueKind::FullPrecision,
    GemmWeightPrologueKind::ScaleBiasDequant,
    GemmWeightPrologueKind::ScaleZeroPointDequant)
VARIANTS(BITS, 0, 4, 8)
VARIANTS(GROUP_SIZE, 0, 32, 64, 128)
CONSTRAINT(
    !USE_MXU || GEMM_TILING == GemmTiling::T64x64x32_2x2 ||
        GEMM_TILING == GemmTiling::T32x64x32_2x2 ||
        GEMM_TILING == GemmTiling::T64x32x32_4x1 ||
        GEMM_TILING == GemmTiling::T128x128x32_4x4)
CONSTRAINT(
    WEIGHT_PROLOGUE != GemmWeightPrologueKind::FullPrecision ||
        (BITS == 0 && GROUP_SIZE == 0))
CONSTRAINT(
    WEIGHT_PROLOGUE == GemmWeightPrologueKind::FullPrecision ||
        (BITS != 0 && GROUP_SIZE != 0))
CONSTRAINT(
    WEIGHT_PROLOGUE == GemmWeightPrologueKind::FullPrecision ||
        (!USE_MXU && TRANSPOSE_B &&
         (GEMM_TILING == GemmTiling::T8x32x32_1x1 ||
          GEMM_TILING == GemmTiling::T32x32x32_2x2 ||
          GEMM_TILING == GemmTiling::T64x32x32_2x2 ||
          GEMM_TILING == GemmTiling::T64x64x32_2x2 ||
          GEMM_TILING == GemmTiling::T64x64x64_2x2)))
CONSTRAINT(
    GEMM_TILING != GemmTiling::T64x64x64_2x2 ||
        GROUP_SIZE == 64 || GROUP_SIZE == 128)
KERNEL(Gemm)(
    const device T* a,
    const device uint8_t* b_packed,
    device T* d,
    const device T* scales
        OPTIONAL(WEIGHT_PROLOGUE != GemmWeightPrologueKind::FullPrecision),
    const device T* biases
        OPTIONAL(WEIGHT_PROLOGUE == GemmWeightPrologueKind::ScaleBiasDequant),
    const device uint8_t* zero_points
        OPTIONAL(WEIGHT_PROLOGUE == GemmWeightPrologueKind::ScaleZeroPointDequant),
    const device T* output_bias
        OPTIONAL(output_transform.contains(GemmDTransform::BIAS)),
    const constant uzu::matmul::GemmParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const GemmDTransform output_transform SPECIALIZE,
    const GemmAlignment alignment SPECIALIZE,
    threadgroup T a_shared[GEMM_TGA_ELEMENTS],
    threadgroup T b_shared[GEMM_TGB_ELEMENTS],
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint thread_x THREADS(METAL_SIMD_SIZE),
    const uint thread_y THREADS(gemm_tiling_simdgroups_per_column(GEMM_TILING)),
    const uint thread_z THREADS(gemm_tiling_simdgroups_per_row(GEMM_TILING)),
    const ThreadContext thread_context
) {
  (void)group_x;
  (void)group_y;
  (void)thread_x;
  (void)thread_y;
  (void)thread_z;

  if constexpr (USE_MXU) {
    (void)scales;
    (void)biases;
    (void)zero_points;
    const device T* b = reinterpret_cast<const device T*>(b_packed);
    MxuMmaCore<T, GEMM_TILING, TRANSPOSE_B>::run(
        a,
        b,
        d,
        params,
        alignment,
        output_transform,
        output_bias,
        thread_context
    );
  } else {
    SimdgroupMmaCore<
        T,
        GEMM_TILING,
        TRANSPOSE_B,
        WEIGHT_PROLOGUE,
        BITS,
        GROUP_SIZE>::
        run(a,
            b_packed,
            d,
            params,
            alignment,
            output_transform,
            scales,
            biases,
            zero_points,
            output_bias,
            a_shared,
            b_shared,
            thread_context);
  }
}
