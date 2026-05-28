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
              (gemm_tiling_block_k(GEMM_TILING) + 16 / int(sizeof(AT)))))
#define GEMM_TGB_ELEMENTS                                                      \
  (USE_MXU ? 1                                                                 \
           : (gemm_tiling_block_n(GEMM_TILING) *                               \
              (gemm_tiling_block_k(GEMM_TILING) + 16 / int(sizeof(BT)))))

template <
    typename AT,
    typename BT,
    typename DT,
    GemmTiling GEMM_TILING,
    bool TRANSPOSE_B,
    bool USE_MXU,
    GemmWeightPrologueKind WEIGHT_PROLOGUE,
    uint BITS,
    uint GROUP_SIZE>
VARIANTS(AT, bfloat, float)
VARIANTS(BT, bfloat, float)
VARIANTS(DT, bfloat, float)
CONSTRAINT(BT != "float" || (AT == "float" && DT == "float"))
VARIANTS(
    GEMM_TILING,
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
    GemmWeightPrologueKind::ScaleZeroPointDequant,
    GemmWeightPrologueKind::ScaleSymmetricDequant)
VARIANTS(BITS, 0, 4, 8)
VARIANTS(GROUP_SIZE, 0, 16, 32, 64, 128)
CONSTRAINT(
    !USE_MXU || GEMM_TILING == GemmTiling::T64x64x32_2x2 ||
        GEMM_TILING == GemmTiling::T32x64x32_2x2 ||
        GEMM_TILING == GemmTiling::T64x32x32_4x1 ||
        GEMM_TILING == GemmTiling::T128x128x32_4x4)
CONSTRAINT(
    WEIGHT_PROLOGUE != GemmWeightPrologueKind::FullPrecision || USE_MXU ||
        GEMM_TILING == GemmTiling::T64x64x16_2x2 ||
        GEMM_TILING == GemmTiling::T64x32x32_2x2)
CONSTRAINT(
    TRANSPOSE_B ||
        (WEIGHT_PROLOGUE == GemmWeightPrologueKind::FullPrecision &&
         ((!USE_MXU &&
           (GEMM_TILING == GemmTiling::T64x64x16_2x2 ||
            GEMM_TILING == GemmTiling::T64x32x32_2x2)) ||
          (USE_MXU &&
           (GEMM_TILING == GemmTiling::T32x64x32_2x2 ||
            GEMM_TILING == GemmTiling::T64x64x32_2x2 ||
            GEMM_TILING == GemmTiling::T128x128x32_4x4)))))
CONSTRAINT((WEIGHT_PROLOGUE == GemmWeightPrologueKind::FullPrecision) == (BITS == 0))
CONSTRAINT((BITS == 0) == (GROUP_SIZE == 0))
CONSTRAINT(WEIGHT_PROLOGUE == GemmWeightPrologueKind::FullPrecision || BT != "float")
CONSTRAINT(
    WEIGHT_PROLOGUE == GemmWeightPrologueKind::FullPrecision ||
        (!USE_MXU && TRANSPOSE_B &&
         (GEMM_TILING == GemmTiling::T8x32x32_1x1 ||
          GEMM_TILING == GemmTiling::T64x64x16_2x2 ||
          GEMM_TILING == GemmTiling::T32x32x32_2x2 ||
          GEMM_TILING == GemmTiling::T64x32x32_2x2 ||
          GEMM_TILING == GemmTiling::T64x64x32_2x2)))
CONSTRAINT(
    WEIGHT_PROLOGUE == GemmWeightPrologueKind::FullPrecision ||
        GROUP_SIZE != 16 || GEMM_TILING == GemmTiling::T64x64x16_2x2)
KERNEL(Gemm)(
    const device AT* a,
    const device uint8_t* b_packed,
    device DT* d,
    const device BT* scales
        OPTIONAL(WEIGHT_PROLOGUE != GemmWeightPrologueKind::FullPrecision),
    const device BT* biases
        OPTIONAL(WEIGHT_PROLOGUE == GemmWeightPrologueKind::ScaleBiasDequant),
    const device uint8_t* zero_points
        OPTIONAL(WEIGHT_PROLOGUE == GemmWeightPrologueKind::ScaleZeroPointDequant),
    const device BT* output_bias
        OPTIONAL(output_transform.contains(GemmDTransform::BIAS)),
    const constant uzu::matmul::GemmParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const GemmDTransform output_transform SPECIALIZE,
    const GemmAlignment alignment SPECIALIZE,
    threadgroup AT a_shared[GEMM_TGA_ELEMENTS],
    threadgroup BT b_shared[GEMM_TGB_ELEMENTS],
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
    const device BT* b = reinterpret_cast<const device BT*>(b_packed);
    MxuMmaCore<AT, BT, DT, GEMM_TILING, TRANSPOSE_B>::run(
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
        AT,
        BT,
        DT,
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
