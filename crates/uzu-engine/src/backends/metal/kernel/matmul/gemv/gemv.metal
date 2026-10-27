#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../../common/integral_constant.h"
#include "../../generated/gemm.h"
#include "common/arguments.h"
#include "common/epilogue.h"
#include "common/full_precision_b_source.h"
#include "common/tile.h"
#include "common/quant_b_source.h"
#include "common/reduce.h"

using namespace metal;
using namespace uzu;
using namespace uzu::gemm;

#define GEMV_BATCHED_INPUT                                                                                             \
  (INPUT_ALIGNED && GROUP_LANES == 1 && B_PROLOGUE != GemmBPrologueKind::ScaleBiasDequant &&                           \
   (BITS == 4 || GROUP_SIZE == 64))
#define GEMV_FULL_PRECISION_TILE                                                                                       \
  (BITS == 0 && NUM_SIMDGROUPS == 8 && REDUCTION_LANES == 32 && GROUP_LANES == 1 && (INPUT_ALIGNED || K_SPLIT == 1) && \
   (OUTPUT_ROW_TILE == 8 / K_SPLIT || OUTPUT_ROW_TILE == 32 / K_SPLIT) && INPUT_ROW_TILE == 1)
#define GEMV_QUANTIZED_ROW_TILE                                                                                        \
  (K_SPLIT == 1 && REDUCTION_LANES == 32 &&                                                                            \
   ((BITS == 4 && GROUP_SIZE == 16 * GROUP_LANES) || (BITS == 8 && GROUP_SIZE == 8 * GROUP_LANES)) &&                  \
   ((NUM_SIMDGROUPS == 8 && OUTPUT_ROW_TILE == 32) ||                                                                  \
    (BITS == 4 &&                                                                                                      \
     ((NUM_SIMDGROUPS == 2 && (OUTPUT_ROW_TILE == 2 || OUTPUT_ROW_TILE == 4 || OUTPUT_ROW_TILE == 8)) ||               \
      (NUM_SIMDGROUPS == 4 && (OUTPUT_ROW_TILE == 8 || OUTPUT_ROW_TILE == 16 || OUTPUT_ROW_TILE == 32)) ||             \
      (NUM_SIMDGROUPS == 8 && OUTPUT_ROW_TILE == 16)))) &&                                                             \
   INPUT_ROW_TILE == 1)
#define GEMV_BATCHED_FMA_TILE                                                                                          \
  (K_SPLIT == 1 && REDUCTION_LANES == 8 && NUM_SIMDGROUPS == 4 && OUTPUT_ROW_TILE == 32 && INPUT_ROW_TILE >= 2 &&      \
   INPUT_ROW_TILE <= 4)
template <
    typename AT,
    typename BT,
    typename DT,
    GemmBPrologueKind B_PROLOGUE,
    uint GROUP_SIZE,
    uint BITS,
    uint K_SPLIT,
    bool INPUT_ALIGNED,
    uint INPUT_ROW_TILE,
    uint OUTPUT_ROW_TILE,
    uint REDUCTION_LANES,
    uint GROUP_LANES,
    uint NUM_SIMDGROUPS>
VARIANTS(AT, bfloat)
VARIANTS(BT, bfloat)
VARIANTS(DT, bfloat, float)
VARIANTS(
    B_PROLOGUE,
    GemmBPrologueKind::FullPrecision,
    GemmBPrologueKind::ScaleBiasDequant,
    GemmBPrologueKind::ScaleZeroPointDequant,
    GemmBPrologueKind::ScaleSymmetricDequant)
VARIANTS(GROUP_SIZE, 0, 32, 64)
VARIANTS(BITS, 0, 4, 8)
VARIANTS(K_SPLIT, 1, 2, 4, 8)
VARIANTS(INPUT_ALIGNED, false, true)
VARIANTS(INPUT_ROW_TILE, 1, 2, 3, 4)
VARIANTS(OUTPUT_ROW_TILE, 1, 2, 4, 8, 16, 32)
VARIANTS(REDUCTION_LANES, 8, 32)
VARIANTS(GROUP_LANES, 1, 2, 4, 8)
VARIANTS(NUM_SIMDGROUPS, 2, 4, 8)

CONSTRAINT(
    (B_PROLOGUE == GemmBPrologueKind::FullPrecision && BITS == 0 && GROUP_SIZE == 0) ||
    ((B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant ||
      B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant) &&
     BITS == 4 && GROUP_SIZE != 0) ||
    (B_PROLOGUE == GemmBPrologueKind::ScaleSymmetricDequant && BITS == 8 && GROUP_SIZE != 0))
CONSTRAINT(INPUT_ROW_TILE == 1 || GEMV_BATCHED_INPUT)
CONSTRAINT(
    GEMV_FULL_PRECISION_TILE || (BITS != 0 && (GEMV_QUANTIZED_ROW_TILE || GEMV_BATCHED_FMA_TILE)))
KERNEL(Gemv)(
    const device uint32_t* b,
    const device BT* scales
        OPTIONAL(B_PROLOGUE != GemmBPrologueKind::FullPrecision),
    const device uint8_t* zero_points
        OPTIONAL(B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant),
    const device BT* biases
        OPTIONAL(B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant),
    const device AT* a,
    device DT* d,
    const device BT* output_bias
        OPTIONAL(output_transform.contains(GemmDTransform::BIAS)),
    const device int32_t* hadamard_factors
        OPTIONAL(output_transform.contains(GemmDTransform::RHT)),
    const device uint* gather_indices OPTIONAL(gathered),
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& batch_size,
    const constant float& ab_scale,
    const constant uint& group_count_x,
    const constant uint& scale_output_stride,
    const constant uint& scale_group_stride,
    const constant uint& zero_point_output_stride,
    const constant uint& zero_point_group_stride,
    const constant float& soft_cap
        OPTIONAL(output_transform.contains(GemmDTransform::SOFT_CAP)),
    const GemmDTransform output_transform SPECIALIZE,
    const bool gathered SPECIALIZE,
    const bool signed_codes SPECIALIZE,
    const GemmBLayout gemm_b_layout SPECIALIZE,
    const bool full_tile SPECIALIZE,
    threadgroup float shared_results[INPUT_ROW_TILE * OUTPUT_ROW_TILE * K_SPLIT],
    const uint input_tile_idx GROUPS(batch_size.div_ceil(INPUT_ROW_TILE)),
    const uint output_tile_idx GROUPS(group_count_x),
    const uint simd_lane THREADS(32),
    const uint simd_group THREADS(NUM_SIMDGROUPS)
) {
  using Ops = GemvOperands<AT, BT, DT>;
  const Ops ops = {b, scales, zero_points, biases, a, d, output_bias, hadamard_factors, gather_indices};
  const GemvParams params = {
      in_vec_size,
      out_vec_size,
      scale_output_stride,
      scale_group_stride,
      zero_point_output_stride,
      zero_point_group_stride,
      batch_size,
      ab_scale,
      soft_cap,
      output_transform,
      gathered,
      signed_codes,
      gemm_b_layout
  };
  dispatch_bool(full_tile, [&](auto full_tile_constant) {
    constexpr bool FullTile = decltype(full_tile_constant)::value;
    using Tile = GemvTile<INPUT_ROW_TILE, OUTPUT_ROW_TILE, REDUCTION_LANES, GROUP_LANES, NUM_SIMDGROUPS, K_SPLIT>;
    const OutputTile<Tile, FullTile> tile =
        OutputTile<Tile, FullTile>::make(output_tile_idx, input_tile_idx, simd_group, simd_lane, out_vec_size);

    thread float result[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE] = {{0}};

    if constexpr (BITS == 0) {
      FullPrecisionBSource<Tile, AT, BT, DT, INPUT_ALIGNED, FullTile>::accumulate(result, ops, params, tile);
    } else {
      QuantBSource<Tile, AT, BT, DT, B_PROLOGUE, GROUP_SIZE, BITS, INPUT_ALIGNED, FullTile>::accumulate(
          result,
          ops,
          params,
          tile
      );
    }

    Reduce<Tile, FullTile>::run(result, shared_results, tile);
    Epilogue<Tile, AT, BT, DT, FullTile>::store(result, ops, params, tile, shared_results);
  });
}
