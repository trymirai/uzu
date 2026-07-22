#include "../../common/dsl.h"
#include "../../common/defines.h"
#include "../../common/thread_context.h"
#include "../generated/gemm.h"

#include "common/mxu_mma_core.h"
#include "common/simdgroup_mma_core.h"

using namespace metal;
using namespace uzu::gemm;

#define GEMM_MXU_QUANT (gemm_tiling_use_mxu(GEMM_TILING) && B_PROLOGUE != GemmBPrologueKind::FullPrecision)
#define GEMM_TGA_ELEMENTS                                                                                              \
  (gemm_tiling_use_mxu(GEMM_TILING)                                                                                    \
       ? 1                                                                                                             \
       : (gemm_tiling_block_m(GEMM_TILING) * (gemm_tiling_block_k(GEMM_TILING) + 16 / int(sizeof(AT)))))
#define GEMM_TGB_ELEMENTS                                                                                              \
  (gemm_tiling_use_mxu(GEMM_TILING)                                                                                    \
       ? (GEMM_MXU_QUANT ? (gemm_tiling_block_n(GEMM_TILING) * (int(GROUP_SIZE) + 16 / int(sizeof(BT)))) : 1)          \
       : (gemm_tiling_block_n(GEMM_TILING) * (gemm_tiling_block_k(GEMM_TILING) + 16 / int(sizeof(BT)))))

template <
    typename AT,
    typename BT,
    typename DT,
    GemmTiling GEMM_TILING,
    bool TRANSPOSE_B,
    GemmBPrologueKind B_PROLOGUE,
    uint BITS,
    uint GROUP_SIZE>
VARIANTS(AT, bfloat, float)
VARIANTS(BT, bfloat, float)
VARIANTS(DT, bfloat, float)
CONSTRAINT(BT != "float" || (AT == "float" && DT == "float"))
VARIANTS(BITS, 0, 4, 8)
VARIANTS(GROUP_SIZE, 0, 16, 32, 64, 128)
CONSTRAINT(B_PROLOGUE == GemmBPrologueKind::FullPrecision || BT != "float")
CONSTRAINT(
    GROUP_SIZE != 16 ||
    GEMM_TILING == GemmTiling::Tile64x64x16_Simdgroups2x2)
CONSTRAINT(
    B_PROLOGUE == GemmBPrologueKind::FullPrecision ||
    (TRANSPOSE_B &&
     (GEMM_TILING != GemmTiling::Tile64x64x16_Simdgroups2x2 ||
      GROUP_SIZE == 16)))
// A 128-wide N block cannot also stage a 128-element quant group in threadgroup memory.
CONSTRAINT(
    B_PROLOGUE == GemmBPrologueKind::FullPrecision ||
    gemm_tiling_block_n(GEMM_TILING) != 128 || GROUP_SIZE <= 64)
// Short-M tiles are only instantiated for the transposed full-precision path.
CONSTRAINT(
    gemm_tiling_block_m(GEMM_TILING) != 16 ||
    (TRANSPOSE_B && B_PROLOGUE == GemmBPrologueKind::FullPrecision))
KERNEL(Gemm)(
    const device AT* a,
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

  if constexpr (gemm_tiling_use_mxu(GEMM_TILING)) {
    MxuMmaCore<AT, BT, DT, GEMM_TILING, TRANSPOSE_B, B_PROLOGUE, BITS, GROUP_SIZE>::run(
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
