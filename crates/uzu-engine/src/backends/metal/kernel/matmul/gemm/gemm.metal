#include "../../common/dsl.h"
#include "../../common/defines.h"
#include "../../common/thread_context.h"
#include "../generated/gemm.h"

#include "common/gemm_tiling.h"
#include "common/mxu_mma_core.h"
#include "common/simdgroup_mma_core.h"
#include "common/register_quant_core.h"
#include "common/packed_w4_input.h"

using namespace metal;
using namespace uzu::gemm;

#define A_IS_INT8 (A_PROLOGUE == GemmAPrologueKind::Int8Symmetric)
#define GEMM_PACKED_W4_A UZU_GEMM_IS_PACKED_W4_TILING(GEMM_TILING)
#define GEMM_REGISTER_QUANT UZU_GEMM_IS_REGISTER_QUANT_TILING(GEMM_TILING)
#define GEMM_REGISTER_W8 (GEMM_REGISTER_QUANT && !GEMM_PACKED_W4_A)
#define NEEDS_ASYMMETRIC_WEIGHT_CORRECTION (A_IS_INT8 && B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant)
#define GEMM_MXU_QUANT (USE_MXU && B_PROLOGUE != GemmBPrologueKind::FullPrecision && !A_IS_INT8)
#define GEMM_TGA_ELEMENTS                                                                                              \
  ((GEMM_REGISTER_QUANT || USE_MXU)                                                                                    \
       ? 1                                                                                                             \
       : (gemm_tiling_block_m(GEMM_TILING) * (gemm_tiling_block_k(GEMM_TILING) + 16 / int(sizeof(AT)))))
#define GEMM_K_PARTIAL_ELEMENTS                                                                                        \
  (gemm_tiling_simdgroups_per_k(GEMM_TILING) == 1                                                                      \
       ? 1                                                                                                             \
       : (gemm_tiling_simdgroups_per_k(GEMM_TILING) - 1) * gemm_tiling_block_m(GEMM_TILING) *                          \
             gemm_tiling_block_n(GEMM_TILING) * uint(sizeof(float) / sizeof(BT)))
// The A8 MXU route reads its scales into registers, so it asks for no
// threadgroup B memory.
#define GEMM_TGB_ELEMENTS                                                                                              \
  ((USE_MXU) ? (GEMM_MXU_QUANT ? (gemm_tiling_block_n(GEMM_TILING) * (int(GROUP_SIZE) + 16 / int(sizeof(BT))))         \
                               : GEMM_K_PARTIAL_ELEMENTS)                                                              \
             : (GEMM_REGISTER_QUANT                                                                                    \
                    ? GEMM_K_PARTIAL_ELEMENTS                                                                          \
                    : gemm_tiling_block_n(GEMM_TILING) * (gemm_tiling_block_k(GEMM_TILING) + 16 / int(sizeof(BT)))))

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
    uint A_GROUP_SIZE,
    bool FLOAT_PARTIALS>
VARIANTS(AT, bfloat)
VARIANTS(BT, bfloat)
VARIANTS(DT, bfloat, float)
VARIANTS(
    GEMM_TILING,
    GemmTiling::Tile8x32x32_Simdgroups1x1,
    GemmTiling::Tile64x32x32_Simdgroups2x2,
    GemmTiling::Tile64x64x16_Simdgroups2x2,
    GemmTiling::Tile64x64x32_Simdgroups2x2,
    GemmTiling::Tile32x32x32_Simdgroups2x2,
    GemmTiling::Tile16x32x256_Simdgroups1x1,
    GemmTiling::Tile16x32x256_Simdgroups1x1x4,
    GemmTiling::Tile16x128x256_Simdgroups1x4,
    GemmTiling::Tile32x64x256_Simdgroups2x2,
    GemmTiling::Tile64x32x256_Simdgroups4x1,
    GemmTiling::Tile64x64x256_Simdgroups2x2,
    GemmTiling::Tile128x128x256_Simdgroups4x4,
    GemmTiling::Tile8x32x64_Simdgroups1x2x4,
    GemmTiling::Tile8x32x128_Simdgroups1x2x4)
VARIANTS(TRANSPOSE_B, false, true)
VARIANTS(USE_MXU, false, true)
VARIANTS(
    B_PROLOGUE,
    GemmBPrologueKind::FullPrecision,
    GemmBPrologueKind::ScaleBiasDequant,
    GemmBPrologueKind::ScaleZeroPointDequant,
    GemmBPrologueKind::ScaleSymmetricDequant)
VARIANTS(BITS, 0, 4, 8)
VARIANTS(GROUP_SIZE, 0, 32, 64)
VARIANTS(
    A_PROLOGUE,
    GemmAPrologueKind::FullPrecision,
    GemmAPrologueKind::Int8Symmetric)
VARIANTS(A_GROUP_SIZE, 0, 128)
VARIANTS(FLOAT_PARTIALS, false, true)
CONSTRAINT((DT == "float") == FLOAT_PARTIALS)
// Float output is limited to int8 accumulation or register-route split-K partials.
CONSTRAINT(!FLOAT_PARTIALS || A_IS_INT8 || GEMM_REGISTER_QUANT)
CONSTRAINT(
    USE_MXU ==
    (GEMM_TILING == GemmTiling::Tile16x32x256_Simdgroups1x1 ||
     GEMM_TILING == GemmTiling::Tile16x32x256_Simdgroups1x1x4 ||
     GEMM_TILING == GemmTiling::Tile16x128x256_Simdgroups1x4 ||
     GEMM_TILING == GemmTiling::Tile32x64x256_Simdgroups2x2 ||
     GEMM_TILING == GemmTiling::Tile64x32x256_Simdgroups4x1 ||
     GEMM_TILING == GemmTiling::Tile64x64x256_Simdgroups2x2 ||
     GEMM_TILING == GemmTiling::Tile128x128x256_Simdgroups4x4))
CONSTRAINT((B_PROLOGUE == GemmBPrologueKind::FullPrecision) == (BITS == 0))
CONSTRAINT((BITS == 0) == (GROUP_SIZE == 0))
CONSTRAINT(B_PROLOGUE != GemmBPrologueKind::ScaleZeroPointDequant || BITS == 4)
CONSTRAINT(B_PROLOGUE != GemmBPrologueKind::ScaleSymmetricDequant || BITS == 8)
CONSTRAINT(B_PROLOGUE != GemmBPrologueKind::ScaleBiasDequant || BITS == 4)
CONSTRAINT(
    B_PROLOGUE == GemmBPrologueKind::FullPrecision ||
    (TRANSPOSE_B &&
     GEMM_TILING != GemmTiling::Tile64x64x16_Simdgroups2x2 &&
     GEMM_TILING != GemmTiling::Tile64x32x32_Simdgroups2x2))
CONSTRAINT(
    B_PROLOGUE != GemmBPrologueKind::FullPrecision || USE_MXU ||
    GEMM_TILING == GemmTiling::Tile64x64x16_Simdgroups2x2 ||
    GEMM_TILING == GemmTiling::Tile64x32x32_Simdgroups2x2)
CONSTRAINT(
    GEMM_TILING != GemmTiling::Tile16x32x256_Simdgroups1x1 ||
    (TRANSPOSE_B &&
     (B_PROLOGUE == GemmBPrologueKind::FullPrecision ||
      A_PROLOGUE == GemmAPrologueKind::Int8Symmetric)))
CONSTRAINT(
    GEMM_TILING != GemmTiling::Tile16x32x256_Simdgroups1x1x4 ||
    A_PROLOGUE == GemmAPrologueKind::Int8Symmetric)
CONSTRAINT(
    GEMM_TILING != GemmTiling::Tile16x128x256_Simdgroups1x4 ||
    (TRANSPOSE_B && B_PROLOGUE == GemmBPrologueKind::FullPrecision))
CONSTRAINT(A_PROLOGUE == GemmAPrologueKind::FullPrecision || USE_MXU)
CONSTRAINT(
    !GEMM_PACKED_W4_A ||
    (!USE_MXU && TRANSPOSE_B && A_PROLOGUE == GemmAPrologueKind::FullPrecision &&
     B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant && BITS == 4 && (GROUP_SIZE == 32 || GROUP_SIZE == 64)))
CONSTRAINT(
    !GEMM_REGISTER_W8 ||
    (!USE_MXU && TRANSPOSE_B && A_PROLOGUE == GemmAPrologueKind::FullPrecision &&
     B_PROLOGUE == GemmBPrologueKind::ScaleSymmetricDequant && BITS == 8 && GROUP_SIZE == 64))
CONSTRAINT(
    A_PROLOGUE == GemmAPrologueKind::FullPrecision ||
    (TRANSPOSE_B && B_PROLOGUE != GemmBPrologueKind::FullPrecision))
CONSTRAINT((A_PROLOGUE == GemmAPrologueKind::FullPrecision) == (A_GROUP_SIZE == 0))
KERNEL(Gemm)(
    const device AT* a OPTIONAL(A_PROLOGUE == GemmAPrologueKind::FullPrecision),
    const device uint* packed_a OPTIONAL(GEMM_PACKED_W4_A),
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
    const constant float& soft_cap
        OPTIONAL(output_transform.contains(GemmDTransform::SOFT_CAP)),
    const device int8_t* a_int8 OPTIONAL(A_IS_INT8),
    const device float* a_scales OPTIONAL(A_IS_INT8),
    const device int32_t* a_group_sums OPTIONAL(NEEDS_ASYMMETRIC_WEIGHT_CORRECTION),
    const constant uzu::matmul::GemmParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant uint& group_count_z,
    const GemmDTransform output_transform SPECIALIZE,
    const GemmAlignment alignment SPECIALIZE,
    const bool signed_codes SPECIALIZE,
    const GemmBLayout gemm_b_layout SPECIALIZE,
    threadgroup AT a_shared[GEMM_TGA_ELEMENTS],
    threadgroup BT b_shared[GEMM_TGB_ELEMENTS],
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint group_z GROUPS(group_count_z),
    const uint thread_x THREADS(METAL_SIMD_SIZE),
    const uint thread_y THREADS(gemm_tiling_simdgroups_per_column(GEMM_TILING)),
    const uint thread_z THREADS(gemm_tiling_simdgroups_per_row(GEMM_TILING) * gemm_tiling_simdgroups_per_k(GEMM_TILING)),
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
  using RightOperand = operands::RightOperandFor<B_PROLOGUE, ushort(BITS), ushort(GROUP_SIZE), BT>;
  static_assert(
      NEEDS_ASYMMETRIC_WEIGHT_CORRECTION == (A_IS_INT8 && RightOperand::NEEDS_CORRECTION),
      "kernel bindings and operand correction policy must agree"
  );
  const auto left_storage = operands::pack_left<LeftOperand, AT>(a, a_int8, a_scales, a_group_sums);
  const auto right_storage =
      operands::pack_right<RightOperand, BT>(b, scales, biases, zero_points, signed_codes, gemm_b_layout);

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
  } else if constexpr (GEMM_REGISTER_QUANT) {
    using Core = RegisterQuantMmaCore<DT, GEMM_TILING, TRANSPOSE_B, LeftOperand, RightOperand>;
    Core::run(
        left_storage,
        right_storage,
        packed_a,
        d,
        params,
        output_transform,
        soft_cap,
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

template <uint GROUP_SIZE>
VARIANTS(GROUP_SIZE, 32, 64)
KERNEL(GemmPackInput)(
    const device bfloat* a,
    device uint* packed,
    const constant uint& in_vec_size,
    const constant uint& batch_size,
    const uint group_index GROUPS(in_vec_size / GROUP_SIZE),
    const uint lane THREADS(32)
) {
  constexpr uint STEPS = PackedW4Input<GROUP_SIZE>::STEPS;
  constexpr uint RUNS = STEPS / 4;
  constexpr uint GROUP_WORDS = PackedW4Input<GROUP_SIZE>::GROUP_WORDS;
  const uint q = lane / 8;
  const uint kk = lane % 8;
  const uint i = kk / 2;
  const uint e = kk % 2;
  const uint row0 = 2 * q;
  const uint row1 = 2 * q + 1;
  // Runs of 4 consecutive positions (one per word of the lane slot), 8 bytes each.
  const uint offset = PackedW4Input<GROUP_SIZE>::activation_offset(group_index, i, e);
  const device uint2* x0 = reinterpret_cast<const device uint2*>(a + min(row0, batch_size - 1) * in_vec_size + offset);
  const device uint2* x1 = reinterpret_cast<const device uint2*>(a + min(row1, batch_size - 1) * in_vec_size + offset);
  const bool v0 = row0 < batch_size;
  const bool v1 = row1 < batch_size;
  device uint* out = packed + group_index * GROUP_WORDS;
  uint2 raw0[RUNS], raw1[RUNS];
  METAL_PRAGMA_UNROLL
  for (uint run = 0; run < RUNS; run++) {
    raw0[run] = v0 ? x0[2 * run] : uint2(0u);
    raw1[run] = v1 ? x1[2 * run] : uint2(0u);
  }
  float t0 = 0.0f, r0 = 0.0f, t1 = 0.0f, r1 = 0.0f;
  uint words[STEPS];
  METAL_PRAGMA_UNROLL
  for (uint s = 0; s < STEPS; s++) {
    const uint j = s % 4;
    const uint word0 = (j / 2 == 0 ? raw0[s / 4].x : raw0[s / 4].y);
    const uint word1 = (j / 2 == 0 ? raw1[s / 4].x : raw1[s / 4].y);
    const uint bits0 = j % 2 == 0 ? word0 << 16 : word0 & 0xffff0000u;
    const uint bits1 = j % 2 == 0 ? word1 << 16 : word1 & 0xffff0000u;
    const half h0 = half(clamp(as_type<float>(bits0), -65504.0f, 65504.0f));
    const half h1 = half(clamp(as_type<float>(bits1), -65504.0f, 65504.0f));
    const float bias = w4_pair_bias(j);
    t0 += float(h0);
    r0 += bias * float(h0);
    t1 += float(h1);
    r1 += bias * float(h1);
    words[s] = as_type<uint>(half2(h0, h1));
  }
  METAL_PRAGMA_UNROLL
  for (uint v = 0; v < STEPS / 4; v++) {
    reinterpret_cast<device uint4*>(out + PackedW4Input<GROUP_SIZE>::pair_offset(q, kk))[v] =
        uint4(words[4 * v], words[4 * v + 1], words[4 * v + 2], words[4 * v + 3]);
  }
  METAL_PRAGMA_UNROLL
  for (uint mask = 1; mask < 8; mask *= 2) {
    t0 += simd_shuffle_xor(t0, mask);
    r0 += simd_shuffle_xor(r0, mask);
    t1 += simd_shuffle_xor(t1, mask);
    r1 += simd_shuffle_xor(r1, mask);
  }
  if (kk == 0) {
    reinterpret_cast<device float4*>(out + PackedW4Input<GROUP_SIZE>::sums_offset())[q] = float4(t0, r0, t1, r1);
  }
}
