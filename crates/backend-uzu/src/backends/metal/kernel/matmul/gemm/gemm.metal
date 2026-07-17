#include "../../common/dsl.h"
#include "../../common/defines.h"
#include "../../common/thread_context.h"
#include "../generated/gemm.h"
#include "../generated/activation_prepare.h"

#include "common/gemm_tiling.h"
#include "common/mxu_mma_core.h"
#include "common/simdgroup_mma_core.h"

using namespace metal;
using namespace uzu::gemm;
using namespace uzu::activation_prepare;

#define A_IS_INT8                                                                                                      \
  (A_PROLOGUE == GemmAPrologueKind::Int8Symmetric || A_PROLOGUE == GemmAPrologueKind::Int8Asymmetric)
#define GEMM_MXU_QUANT                                                                                                 \
  (USE_MXU && B_PROLOGUE != GemmBPrologueKind::FullPrecision && !A_IS_INT8)
#define GEMM_TGA_ELEMENTS                                                                                              \
  ((USE_MXU) ? 1 : (gemm_tiling_block_m(GEMM_TILING) * (gemm_tiling_block_k(GEMM_TILING) + 16 / int(sizeof(AT)))))
#define GEMM_TGB_ELEMENTS                                                                                              \
  ((USE_MXU) ? (GEMM_MXU_QUANT ? (gemm_tiling_block_n(GEMM_TILING) * (int(GROUP_SIZE) + 16 / int(sizeof(BT)))) : 1)    \
             : (gemm_tiling_block_n(GEMM_TILING) * (gemm_tiling_block_k(GEMM_TILING) + 16 / int(sizeof(BT)))))

template <
    typename AT,
    typename BT,
    typename DT,
    GemmTiling GEMM_TILING,
    bool TRANSPOSE_B,
    bool USE_MXU,
    GemmBPrologueKind B_PROLOGUE,
    uint BITS,
    uint GROUP_SIZE,
    GemmAPrologueKind A_PROLOGUE>
VARIANTS(AT, bfloat, float)
VARIANTS(BT, bfloat, float)
VARIANTS(DT, bfloat, float)
CONSTRAINT(BT != "float" || (AT == "float" && DT == "float"))
VARIANTS(
    GEMM_TILING,
    GemmTiling::Tile8x32x32_Simdgroups1x1,
    GemmTiling::Tile64x32x32_Simdgroups2x2,
    GemmTiling::Tile64x64x16_Simdgroups2x2,
    GemmTiling::Tile64x64x32_Simdgroups2x2,
    GemmTiling::Tile32x32x32_Simdgroups2x2,
    GemmTiling::Tile16x32x256_Simdgroups1x1,
    GemmTiling::Tile16x128x256_Simdgroups1x4,
    GemmTiling::Tile32x64x256_Simdgroups2x2,
    GemmTiling::Tile64x32x256_Simdgroups4x1,
    GemmTiling::Tile64x64x256_Simdgroups2x2,
    GemmTiling::Tile128x128x256_Simdgroups4x4)
VARIANTS(TRANSPOSE_B, false, true)
VARIANTS(USE_MXU, false, true)
VARIANTS(
    B_PROLOGUE,
    GemmBPrologueKind::FullPrecision,
    GemmBPrologueKind::ScaleBiasDequant,
    GemmBPrologueKind::ScaleZeroPointDequant,
    GemmBPrologueKind::ScaleSymmetricDequant)
VARIANTS(BITS, 0, 4, 8)
VARIANTS(GROUP_SIZE, 0, 16, 32, 64, 128)
VARIANTS(
    A_PROLOGUE,
    GemmAPrologueKind::FullPrecision,
    GemmAPrologueKind::Int8Symmetric,
    GemmAPrologueKind::Int8Asymmetric)
CONSTRAINT(
    USE_MXU ==
    (GEMM_TILING == GemmTiling::Tile16x32x256_Simdgroups1x1 ||
     GEMM_TILING == GemmTiling::Tile16x128x256_Simdgroups1x4 ||
     GEMM_TILING == GemmTiling::Tile32x64x256_Simdgroups2x2 ||
     GEMM_TILING == GemmTiling::Tile64x32x256_Simdgroups4x1 ||
     GEMM_TILING == GemmTiling::Tile64x64x256_Simdgroups2x2 ||
     GEMM_TILING == GemmTiling::Tile128x128x256_Simdgroups4x4))
CONSTRAINT((B_PROLOGUE == GemmBPrologueKind::FullPrecision) == (BITS == 0))
CONSTRAINT((BITS == 0) == (GROUP_SIZE == 0))
CONSTRAINT(B_PROLOGUE == GemmBPrologueKind::FullPrecision || BT != "float")
CONSTRAINT(
    GROUP_SIZE != 16 ||
    GEMM_TILING == GemmTiling::Tile64x64x16_Simdgroups2x2)
CONSTRAINT(
    B_PROLOGUE == GemmBPrologueKind::FullPrecision ||
    (TRANSPOSE_B &&
     (GEMM_TILING != GemmTiling::Tile64x64x16_Simdgroups2x2 ||
      GROUP_SIZE == 16)))
CONSTRAINT(
    B_PROLOGUE == GemmBPrologueKind::FullPrecision ||
    GEMM_TILING != GemmTiling::Tile128x128x256_Simdgroups4x4 ||
    GROUP_SIZE <= 64)
CONSTRAINT(
    !(GEMM_TILING == GemmTiling::Tile16x32x256_Simdgroups1x1 ||
      GEMM_TILING == GemmTiling::Tile16x128x256_Simdgroups1x4) ||
    (TRANSPOSE_B && B_PROLOGUE == GemmBPrologueKind::FullPrecision))
CONSTRAINT(A_PROLOGUE == GemmAPrologueKind::FullPrecision || USE_MXU)
CONSTRAINT(A_PROLOGUE == GemmAPrologueKind::FullPrecision || BITS == 8)
CONSTRAINT(
    A_PROLOGUE == GemmAPrologueKind::FullPrecision ||
    (TRANSPOSE_B &&
     (B_PROLOGUE == GemmBPrologueKind::ScaleSymmetricDequant ||
      B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant)))
CONSTRAINT(A_PROLOGUE == GemmAPrologueKind::FullPrecision || (AT == "bfloat" && DT == "bfloat"))
KERNEL(Gemm)(
    const device AT* a OPTIONAL(A_PROLOGUE == GemmAPrologueKind::FullPrecision),
    const device BT* b,
    device DT* d,
    const device BT* scales
        OPTIONAL(B_PROLOGUE != GemmBPrologueKind::FullPrecision),
    const device BT* biases
        OPTIONAL(B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant),
    const device uint8_t* zero_points
        OPTIONAL(B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant),
    const device BT* output_bias
        OPTIONAL(output_transform.contains(GemmDTransform::BIAS)),
    const device int32_t* rht_factors
        OPTIONAL(output_transform.contains(GemmDTransform::RHT)),
    const device int8_t* a_int8 OPTIONAL(A_IS_INT8),
    const device float* a_scales OPTIONAL(A_IS_INT8),
    const device int8_t* a_zero_points
        OPTIONAL(A_PROLOGUE == GemmAPrologueKind::Int8Asymmetric),
    const device int32_t* a_row_sums
        OPTIONAL(A_IS_INT8 && B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant),
    const device int32_t* b_col_sums
        OPTIONAL(A_PROLOGUE == GemmAPrologueKind::Int8Asymmetric),
    const constant uzu::matmul::GemmParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant uint& group_count_z,
    const GemmDTransform output_transform SPECIALIZE,
    const GemmAlignment alignment SPECIALIZE,
    threadgroup AT a_shared[GEMM_TGA_ELEMENTS],
    threadgroup BT b_shared[GEMM_TGB_ELEMENTS],
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint group_z GROUPS(group_count_z),
    const uint thread_x THREADS(METAL_SIMD_SIZE),
    const uint thread_y THREADS(gemm_tiling_simdgroups_per_column(GEMM_TILING)),
    const uint thread_z THREADS(gemm_tiling_simdgroups_per_row(GEMM_TILING)),
    const ThreadContext thread_context
) {
  (void)group_x;
  (void)group_y;
  (void)group_z;
  (void)thread_x;
  (void)thread_y;
  (void)thread_z;

  if constexpr (USE_MXU) {
    MxuMmaCore<AT, BT, DT, GEMM_TILING, TRANSPOSE_B, B_PROLOGUE, BITS, GROUP_SIZE, A_PROLOGUE>::run(
        a,
        b,
        d,
        params,
        alignment,
        output_transform,
        scales,
        biases,
        zero_points,
        output_bias,
        rht_factors,
        a_int8,
        a_scales,
        a_zero_points,
        a_row_sums,
        b_col_sums,
        b_shared,
        thread_context
    );
  } else {
    SimdgroupMmaCore<AT, BT, DT, GEMM_TILING, TRANSPOSE_B, B_PROLOGUE, BITS, GROUP_SIZE>::run(
        a,
        b,
        d,
        params,
        alignment,
        output_transform,
        scales,
        biases,
        zero_points,
        output_bias,
        rht_factors,
        a_shared,
        b_shared,
        thread_context
    );
  }
}
