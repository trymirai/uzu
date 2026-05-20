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
           : (gemm_tiling_bm(TILE) *                                             \
              (gemm_tiling_bk(TILE) + 16 / int(sizeof(T)))))
#define GEMM_TGB_ELEMENTS                                                      \
  (USE_MXU ? 1                                                                 \
           : (gemm_tiling_bn(TILE) *                                             \
              (gemm_tiling_bk(TILE) + 16 / int(sizeof(T)))))

template <
    typename T,
    GemmTiling TILE,
    bool TRANSPOSE_B,
    bool USE_MXU,
    GemmWeightPrologueKind WEIGHT_PROLOGUE,
    uint BITS,
    uint GROUP_SIZE>
VARIANTS(T, half, bfloat)
VARIANTS(
    TILE,
    GemmTiling::T8x32x32_1x1,
    GemmTiling::T64x32x32_2x2,
    GemmTiling::T64x64x16_2x2,
    GemmTiling::T64x64x32_2x2,
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
    !USE_MXU || TILE == GemmTiling::T64x64x32_2x2 ||
        TILE == GemmTiling::T32x64x32_2x2 ||
        TILE == GemmTiling::T64x32x32_4x1 ||
        TILE == GemmTiling::T128x128x32_4x4)
CONSTRAINT(
    WEIGHT_PROLOGUE != GemmWeightPrologueKind::FullPrecision ||
        (BITS == 0 && GROUP_SIZE == 0))
CONSTRAINT(
    WEIGHT_PROLOGUE == GemmWeightPrologueKind::FullPrecision ||
        (BITS != 0 && GROUP_SIZE != 0))
CONSTRAINT(
    WEIGHT_PROLOGUE == GemmWeightPrologueKind::FullPrecision ||
        (!USE_MXU && TRANSPOSE_B &&
         (TILE == GemmTiling::T8x32x32_1x1 ||
          TILE == GemmTiling::T32x32x32_2x2 ||
          TILE == GemmTiling::T64x64x32_2x2)))
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
    const constant uzu::matmul::GemmParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const GemmInputPrologueKind input_prologue SPECIALIZE,
    const GemmOutputTransformKind output_transform SPECIALIZE,
    const GemmAlignment alignment SPECIALIZE,
    threadgroup T a_shared[GEMM_TGA_ELEMENTS],
    threadgroup T b_shared[GEMM_TGB_ELEMENTS],
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint thread_x THREADS(METAL_SIMD_SIZE),
    const uint thread_y THREADS(gemm_tiling_smg_n(TILE)),
    const uint thread_z THREADS(gemm_tiling_smg_m(TILE)),
    const ThreadContext thread_context
) {
  (void)group_x;
  (void)group_y;
  (void)thread_x;
  (void)thread_y;
  (void)thread_z;
  (void)input_prologue;
  if (input_prologue != GemmInputPrologueKind::FullPrecision) {
    return;
  }

  if constexpr (USE_MXU) {
    (void)scales;
    (void)biases;
    (void)zero_points;
    const device T* b = reinterpret_cast<const device T*>(b_packed);
    MxuMmaCore<T, TILE, TRANSPOSE_B>::run(
        a, b, d, params, alignment, output_transform, thread_context
    );
  } else {
    (void)scales;
    (void)biases;
    (void)zero_points;
    SimdgroupMmaCore<
        T,
        TILE,
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
            a_shared,
            b_shared,
            thread_context);
  }
}
