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
#define GEMM_TRELLIS (B_PROLOGUE == GemmBPrologueKind::Trellis)
// Trellis weights are symmetric like ScaleSymmetricDequant: the decode emits
// signed levels around zero, so there is no zero point to correct for.
#define NEEDS_ASYMMETRIC_WEIGHT_CORRECTION                                                                             \
  (A_IS_INT8 && !GEMM_TRELLIS && B_PROLOGUE != GemmBPrologueKind::ScaleSymmetricDequant)
#define GEMM_MXU_QUANT (USE_MXU && B_PROLOGUE != GemmBPrologueKind::FullPrecision && !A_IS_INT8)
#define GEMM_TGA_ELEMENTS                                                                                              \
  ((USE_MXU) ? 1 : (gemm_tiling_block_m(GEMM_TILING) * (gemm_tiling_block_k(GEMM_TILING) + 16 / int(sizeof(AT)))))
#define GEMM_INTEGER_TGB_ELEMENTS                                                                                      \
  (GEMM_TRELLIS ? 1                                                                                                    \
                : ((B_PROLOGUE == GemmBPrologueKind::ScaleSymmetricDequant)                                            \
                       ? (2 * gemm_tiling_block_n(GEMM_TILING))                                                        \
                       : (2 * gemm_tiling_block_n(GEMM_TILING) * (1 + 4 / int(sizeof(BT))))))
// The decoded trellis tile: BLOCK_N int8 rows of GROUP_SIZE columns, padded the
// way `MxuMmaCore::SHARED_STRIDE_B` pads them, addressed as words because the
// decode stores four weights at a time. It cannot share `b_shared`: that array
// is `BT`-typed, so it is only 2-byte aligned and the 32-bit stores would fault.
#define GEMM_TRELLIS_TG_WORDS (GEMM_TRELLIS ? (gemm_tiling_block_n(GEMM_TILING) * (int(GROUP_SIZE) + 16) / 4) : 1)
#define GEMM_TGB_ELEMENTS                                                                                              \
  ((USE_MXU) ? (GEMM_MXU_QUANT ? (gemm_tiling_block_n(GEMM_TILING) * (int(GROUP_SIZE) + 16 / int(sizeof(BT))))         \
                               : (A_IS_INT8 ? GEMM_INTEGER_TGB_ELEMENTS : 1))                                          \
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
    GemmAPrologueKind A_PROLOGUE,
    uint A_GROUP_SIZE>
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
    GemmBPrologueKind::ScaleSymmetricDequant,
    GemmBPrologueKind::Trellis)
VARIANTS(BITS, 0, 4, 8)
VARIANTS(GROUP_SIZE, 0, 16, 32, 64, 128)
VARIANTS(
    A_PROLOGUE,
    GemmAPrologueKind::FullPrecision,
    GemmAPrologueKind::Int8Symmetric)
VARIANTS(A_GROUP_SIZE, 0, 32, 64, 128)
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
    (TRANSPOSE_B &&
     (B_PROLOGUE == GemmBPrologueKind::FullPrecision ||
      A_PROLOGUE == GemmAPrologueKind::Int8Symmetric)))
CONSTRAINT(A_PROLOGUE == GemmAPrologueKind::FullPrecision || USE_MXU)
CONSTRAINT(A_PROLOGUE == GemmAPrologueKind::FullPrecision || BITS == 4 || BITS == 8)
CONSTRAINT(
    A_PROLOGUE == GemmAPrologueKind::FullPrecision ||
    (GROUP_SIZE % METAL_SIMD_SIZE == 0 && GROUP_SIZE != 0))
CONSTRAINT(
    A_PROLOGUE == GemmAPrologueKind::FullPrecision ||
    (TRANSPOSE_B && B_PROLOGUE != GemmBPrologueKind::FullPrecision))
CONSTRAINT(A_PROLOGUE == GemmAPrologueKind::FullPrecision || (AT == "bfloat" && DT == "bfloat"))
CONSTRAINT((A_PROLOGUE == GemmAPrologueKind::FullPrecision) == (A_GROUP_SIZE == 0))
CONSTRAINT(A_PROLOGUE == GemmAPrologueKind::FullPrecision || A_GROUP_SIZE >= 32)
// A trellis tape is decoded into threadgroup memory as int8 and fed to the
// integer schedule, so it is the a8 path with one substitution: `GROUP_SIZE` is
// the STAGING block K, not a quantization group, and `BITS` is the width of the
// decode's output. Only the tiles the a8 policy actually selects are compiled.
//
// `GROUP_SIZE == 64` here must match `trellis_format::TRELLIS_BLOCK_K`, which is
// what the host reports as the weight group size; a mismatch is a missing entry
// point at dispatch time, not a compile error. `A_GROUP_SIZE == 128` is
// `ACTIVATION_SCALE_GROUP_SIZE`, the only activation group the a8 route uses.
CONSTRAINT(
    B_PROLOGUE != GemmBPrologueKind::Trellis ||
    (BITS == 8 && GROUP_SIZE == 64 &&
     A_PROLOGUE == GemmAPrologueKind::Int8Symmetric &&
     A_GROUP_SIZE == 128 &&
     (GEMM_TILING == GemmTiling::Tile16x32x256_Simdgroups1x1 ||
      GEMM_TILING == GemmTiling::Tile32x64x256_Simdgroups2x2 ||
      GEMM_TILING == GemmTiling::Tile64x64x256_Simdgroups2x2)))
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
    const device int32_t* a_group_sums OPTIONAL(NEEDS_ASYMMETRIC_WEIGHT_CORRECTION),
    const constant uzu::matmul::TrellisParams& trellis OPTIONAL(GEMM_TRELLIS),
    const constant uzu::matmul::GemmParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant uint& group_count_z,
    const GemmDTransform output_transform SPECIALIZE,
    const GemmAlignment alignment SPECIALIZE,
    const bool signed_codes SPECIALIZE,
    const bool stage_weight_scales SPECIALIZE,
    const bool hoist_operand_addressing SPECIALIZE,
    threadgroup AT a_shared[GEMM_TGA_ELEMENTS],
    threadgroup BT b_shared[GEMM_TGB_ELEMENTS],
    threadgroup uint b_trellis[GEMM_TRELLIS_TG_WORDS],
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

  using LeftOperand = operands::LeftOperandFor<A_PROLOGUE, AT, ushort(A_GROUP_SIZE)>;
  using RightOperand = operands::RightOperandFor<B_PROLOGUE, ushort(BITS), ushort(GROUP_SIZE), BT>;
  static_assert(
      NEEDS_ASYMMETRIC_WEIGHT_CORRECTION == (A_IS_INT8 && RightOperand::NEEDS_CORRECTION),
      "kernel bindings and operand correction policy must agree"
  );
  const auto left_storage = operands::pack_left<LeftOperand, AT>(a, a_int8, a_scales, a_group_sums);
  const auto right_storage =
      operands::pack_right<RightOperand, BT>(b, scales, biases, zero_points, &trellis, signed_codes);

  static_assert(
      !A_IS_INT8 || GEMM_TGB_ELEMENTS >= GEMM_INTEGER_TGB_ELEMENTS,
      "threadgroup scale-line cache is undersized"
  );

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
        operands::stage_block<RightOperand>(b_shared, b_trellis),
        stage_weight_scales,
        hoist_operand_addressing,
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
