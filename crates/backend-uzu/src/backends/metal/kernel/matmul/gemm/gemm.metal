#include "../../common/dsl.h"
#include "../../common/defines.h"
#include "../../common/thread_context.h"
#include "../generated/gemm.h"

#include "common/mxu_mma_core.h"
#include "common/simdgroup_mma_core.h"

using namespace metal;
using namespace uzu::gemm;

// Unqualified aliases for the `GemmWeightPrologueKind` enum so VARIANTS values
// stay valid C identifiers (no `::` in the mangled kernel name) and so the
// `to_string()` of `GemmWeightPrologueKind` (host side) matches the literal
// VARIANT text exactly.
namespace {
constant constexpr GemmWeightPrologueKind FullPrecision =
    GemmWeightPrologueKind::FullPrecision;
constant constexpr GemmWeightPrologueKind ScaleBiasDequant =
    GemmWeightPrologueKind::ScaleBiasDequant;
constant constexpr GemmWeightPrologueKind ScaleZeroPointDequant =
    GemmWeightPrologueKind::ScaleZeroPointDequant;
} // namespace

// Shared array size for the SIMDGROUP path: `BLOCK_M * (BLOCK_K + padding_T)`
// (and similar for B). Padding matches `SimdgroupMmaCore::PADDING_*` =
// 16/sizeof(T). The MXU path doesn't use these arrays, so we collapse them to
// one element to avoid wasting threadgroup memory on big MXU tiles.
#define GEMM_TGA_ELEMENTS                                                      \
  (USE_MXU                                                                     \
       ? 1                                                                     \
       : (THREADGROUP_BLOCK_M * (THREADGROUP_BLOCK_K + 16 / int(sizeof(T)))))
#define GEMM_TGB_ELEMENTS                                                      \
  (USE_MXU                                                                     \
       ? 1                                                                     \
       : (THREADGROUP_BLOCK_N * (THREADGROUP_BLOCK_K + 16 / int(sizeof(T)))))

template <
    typename T,
    uint THREADGROUP_BLOCK_M,
    uint THREADGROUP_BLOCK_N,
    uint THREADGROUP_BLOCK_K,
    uint SIMDGROUPS_M,
    uint SIMDGROUPS_N,
    bool TRANSPOSE_B,
    bool USE_MXU,
    GemmWeightPrologueKind WEIGHT_PROLOGUE,
    uint BITS,
    uint GROUP_SIZE>
VARIANTS(T, float, half, bfloat)
VARIANTS(THREADGROUP_BLOCK_M, 32, 64, 128)
VARIANTS(THREADGROUP_BLOCK_N, 32, 64, 128)
VARIANTS(THREADGROUP_BLOCK_K, 16, 32)
VARIANTS(SIMDGROUPS_M, 1, 2, 4)
VARIANTS(SIMDGROUPS_N, 1, 2, 4)
VARIANTS(TRANSPOSE_B, false, true)
VARIANTS(USE_MXU, false, true)
VARIANTS(
    WEIGHT_PROLOGUE,
    FullPrecision,
    ScaleBiasDequant,
    ScaleZeroPointDequant)
VARIANTS(BITS, 0, 4, 8)
VARIANTS(GROUP_SIZE, 0, 32, 64, 128)
CONSTRAINT(!USE_MXU || THREADGROUP_BLOCK_M % SIMDGROUPS_M == 0)
CONSTRAINT(!USE_MXU || THREADGROUP_BLOCK_N % SIMDGROUPS_N == 0)
CONSTRAINT(!USE_MXU || (THREADGROUP_BLOCK_M / SIMDGROUPS_M) % 16 == 0)
CONSTRAINT(!USE_MXU || (THREADGROUP_BLOCK_N / SIMDGROUPS_N) % 16 == 0)
CONSTRAINT(USE_MXU || max(THREADGROUP_BLOCK_M, THREADGROUP_BLOCK_N) <= 32 * SIMDGROUPS_M * SIMDGROUPS_N)
// Constraint RHS strings match the variant value literals (the build script
// evaluates these in Rhai, where `WEIGHT_PROLOGUE` is bound to the raw text).
//
// FP variants have BITS=0 and GROUP_SIZE=0; quant variants must set both.
CONSTRAINT(
    WEIGHT_PROLOGUE != "FullPrecision" || (BITS == 0 && GROUP_SIZE == 0))
CONSTRAINT(
    WEIGHT_PROLOGUE == "FullPrecision" || (BITS != 0 && GROUP_SIZE != 0))
// Quant only flows through the SIMDGROUP+transposed path.
CONSTRAINT(
    WEIGHT_PROLOGUE == "FullPrecision" || (!USE_MXU && TRANSPOSE_B))
// Loader static_asserts require BCOLS=BK <= group_size.
CONSTRAINT(
    WEIGHT_PROLOGUE == "FullPrecision" || THREADGROUP_BLOCK_K <= GROUP_SIZE)
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
    const uint thread_y THREADS(SIMDGROUPS_N),
    const uint thread_z THREADS(SIMDGROUPS_M),
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
    MxuMmaCore<
        T,
        THREADGROUP_BLOCK_M,
        THREADGROUP_BLOCK_N,
        SIMDGROUPS_M,
        SIMDGROUPS_N,
        TRANSPOSE_B>::
        run(a, b, d, params, alignment, output_transform, thread_context);
  } else {
    if constexpr (WEIGHT_PROLOGUE == GemmWeightPrologueKind::FullPrecision) {
      (void)scales;
      (void)biases;
      (void)zero_points;
    } else if constexpr (
        WEIGHT_PROLOGUE == GemmWeightPrologueKind::ScaleBiasDequant
    ) {
      (void)zero_points;
    } else {
      (void)biases;
    }
    SimdgroupMmaCore<
        T,
        THREADGROUP_BLOCK_M,
        THREADGROUP_BLOCK_N,
        THREADGROUP_BLOCK_K,
        SIMDGROUPS_M,
        SIMDGROUPS_N,
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
