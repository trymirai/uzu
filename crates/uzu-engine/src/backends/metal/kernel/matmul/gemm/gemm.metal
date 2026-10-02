#include "../../common/dsl.h"
#include "../../common/defines.h"
#include "../../common/thread_context.h"
#include "../generated/gemm.h"

#include "common/gemm_tiling.h"
#include "common/mxu_mma_core.h"
#include "common/simdgroup_mma_core.h"

using namespace metal;
using namespace uzu::gemm;

#define A_IS_INT8 (A_PROLOGUE == GemmAPrologueKind::Int8Symmetric)
#define NEEDS_ASYMMETRIC_WEIGHT_CORRECTION (A_IS_INT8 && INTEGER_B_PROLOGUE != GemmBPrologueKind::ScaleSymmetricDequant)
#define GEMM_MXU_QUANT (USE_MXU && BITS != 0 && !A_IS_INT8)
#define GEMM_TGA_ELEMENTS                                                                                              \
  ((USE_MXU) ? 1 : (gemm_tiling_block_m(GEMM_TILING) * (gemm_tiling_block_k(GEMM_TILING) + 16 / int(sizeof(AT)))))
// The A8 MXU route reads its scales into registers, so it asks for no
// threadgroup B memory.
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
    GemmBPrologueKind INTEGER_B_PROLOGUE,
    uint BITS,
    uint GROUP_SIZE,
    GemmAPrologueKind A_PROLOGUE,
    uint A_GROUP_SIZE>
VARIANTS(AT, bfloat, float)
VARIANTS(BT, bfloat)
VARIANTS(DT, bfloat, float)
CONSTRAINT(AT == "bfloat" || DT == "bfloat")
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
CONSTRAINT(TRANSPOSE_B || (AT == "bfloat" && DT == "bfloat"))
VARIANTS(USE_MXU, false, true)
VARIANTS(
    INTEGER_B_PROLOGUE,
    GemmBPrologueKind::FullPrecision,
    GemmBPrologueKind::ScaleBiasDequant,
    GemmBPrologueKind::ScaleZeroPointDequant,
    GemmBPrologueKind::ScaleSymmetricDequant)
VARIANTS(BITS, 0, 4, 8)
VARIANTS(GROUP_SIZE, 0, 16, 32, 64, 128)
VARIANTS(
    A_PROLOGUE,
    GemmAPrologueKind::FullPrecision,
    GemmAPrologueKind::Int8Symmetric)
VARIANTS(A_GROUP_SIZE, 0, 128)
CONSTRAINT(
    USE_MXU ==
    (GEMM_TILING == GemmTiling::Tile16x32x256_Simdgroups1x1 ||
     GEMM_TILING == GemmTiling::Tile16x128x256_Simdgroups1x4 ||
     GEMM_TILING == GemmTiling::Tile32x64x256_Simdgroups2x2 ||
     GEMM_TILING == GemmTiling::Tile64x32x256_Simdgroups4x1 ||
     GEMM_TILING == GemmTiling::Tile64x64x256_Simdgroups2x2 ||
     GEMM_TILING == GemmTiling::Tile128x128x256_Simdgroups4x4))
// Integer MMA retains its scheme-specific code origin and correction arithmetic.
// Dense activations share a loader specialized by the b_prologue function constant.
CONSTRAINT(
    (A_PROLOGUE == GemmAPrologueKind::FullPrecision) ==
    (INTEGER_B_PROLOGUE == GemmBPrologueKind::FullPrecision))
CONSTRAINT((BITS == 0) == (GROUP_SIZE == 0))
CONSTRAINT(
    GROUP_SIZE != 16 ||
    GEMM_TILING == GemmTiling::Tile64x64x16_Simdgroups2x2)
CONSTRAINT(
    BITS == 0 ||
    (TRANSPOSE_B &&
     (GEMM_TILING != GemmTiling::Tile64x64x16_Simdgroups2x2 ||
      GROUP_SIZE == 16)))
CONSTRAINT(
    BITS == 0 ||
    GEMM_TILING != GemmTiling::Tile128x128x256_Simdgroups4x4 ||
    GROUP_SIZE <= 64)
CONSTRAINT(
    !(GEMM_TILING == GemmTiling::Tile16x32x256_Simdgroups1x1 ||
      GEMM_TILING == GemmTiling::Tile16x128x256_Simdgroups1x4) ||
    (TRANSPOSE_B &&
     (BITS == 0 ||
      A_PROLOGUE == GemmAPrologueKind::Int8Symmetric)))
CONSTRAINT(A_PROLOGUE == GemmAPrologueKind::FullPrecision || USE_MXU)
CONSTRAINT(A_PROLOGUE == GemmAPrologueKind::FullPrecision || BITS == 4 || BITS == 8)
CONSTRAINT(
    A_PROLOGUE == GemmAPrologueKind::FullPrecision ||
    (GROUP_SIZE % METAL_SIMD_SIZE == 0 && GROUP_SIZE != 0))
CONSTRAINT(
    A_PROLOGUE == GemmAPrologueKind::FullPrecision ||
    (TRANSPOSE_B && BITS != 0))
CONSTRAINT(A_PROLOGUE == GemmAPrologueKind::FullPrecision || (AT == "bfloat" && DT == "bfloat"))
CONSTRAINT((A_PROLOGUE == GemmAPrologueKind::FullPrecision) == (A_GROUP_SIZE == 0))
// Match the full-precision and quantized selectors' distinct SIMD tile sets.
CONSTRAINT(
    USE_MXU || BITS != 0 ||
    GEMM_TILING == GemmTiling::Tile64x32x32_Simdgroups2x2 ||
    GEMM_TILING == GemmTiling::Tile64x64x16_Simdgroups2x2)
CONSTRAINT(
    BITS == 0 ||
    GEMM_TILING != GemmTiling::Tile64x32x32_Simdgroups2x2)
CONSTRAINT(
    A_PROLOGUE == GemmAPrologueKind::FullPrecision ||
    GEMM_TILING != GemmTiling::Tile16x128x256_Simdgroups1x4)
// The integer schedule drains one weight group at a time, so an activation
// group narrower than the weight group has no kernel.
CONSTRAINT(A_PROLOGUE == GemmAPrologueKind::FullPrecision || A_GROUP_SIZE >= GROUP_SIZE)
KERNEL(Gemm)(
    const device AT* a OPTIONAL(A_PROLOGUE == GemmAPrologueKind::FullPrecision),
    const device BT* b,
    device DT* d,
    const device BT* scales
        OPTIONAL(BITS != 0),
    const device BT* biases
        OPTIONAL(b_prologue == GemmBPrologueKind::ScaleBiasDequant),
    const device uint8_t* zero_points
        OPTIONAL(b_prologue == GemmBPrologueKind::ScaleZeroPointDequant),
    const device BT* output_bias
        OPTIONAL(output_transform.contains(GemmDTransform::BIAS)),
    const device int32_t* rht_factors
        OPTIONAL(output_transform.contains(GemmDTransform::RHT)),
    const device int8_t* a_int8 OPTIONAL(A_IS_INT8),
    const device float* a_scales OPTIONAL(A_IS_INT8),
    const device int32_t* a_group_sums OPTIONAL(NEEDS_ASYMMETRIC_WEIGHT_CORRECTION),
    const constant uzu::matmul::GemmParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant uint& group_count_z,
    const GemmBPrologueKind b_prologue SPECIALIZE,
    const GemmDTransform output_transform SPECIALIZE,
    const GemmAlignment alignment SPECIALIZE,
    const bool signed_codes SPECIALIZE,
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

  using LeftOperand = operands::
      LeftOperandFor<A_PROLOGUE, AT, ushort(A_GROUP_SIZE), A_PROLOGUE == GemmAPrologueKind::Int8Symmetric && BITS == 4>;
  constexpr GemmBPrologueKind OperandScheme = BITS == 0 ? GemmBPrologueKind::FullPrecision
      : A_IS_INT8 ? INTEGER_B_PROLOGUE : GemmBPrologueKind::ScaleSymmetricDequant;
  using RightOperand = operands::RightOperandFor<OperandScheme, ushort(BITS), ushort(GROUP_SIZE), BT>;
  static_assert(
      NEEDS_ASYMMETRIC_WEIGHT_CORRECTION == (A_IS_INT8 && RightOperand::NEEDS_CORRECTION),
      "kernel bindings and operand correction policy must agree"
  );
  const auto left_storage = operands::pack_left<LeftOperand, AT>(a, a_int8, a_scales, a_group_sums);
  const auto right_storage =
      operands::pack_right<RightOperand, BT>(b, scales, biases, zero_points, signed_codes, b_prologue);

  if constexpr (USE_MXU) {
    using Core = MxuMmaCore<DT, GEMM_TILING, TRANSPOSE_B, LeftOperand, RightOperand>;
    Core::run(
        left_storage,
        right_storage,
        d,
        params,
        alignment,
        output_transform,
        output_bias,
        rht_factors,
        b_shared,
        thread_context
    );
  } else {
    using Core = SimdgroupMmaCore<DT, GEMM_TILING, TRANSPOSE_B, LeftOperand, RightOperand>;
    Core::run(
        left_storage,
        right_storage,
        d,
        params,
        alignment,
        output_transform,
        output_bias,
        rht_factors,
        a_shared,
        b_shared,
        thread_context
    );
  }
}
