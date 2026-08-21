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
VARIANTS(AT, bfloat, float)
VARIANTS(BT, bfloat, float)
VARIANTS(DT, bfloat, float)
CONSTRAINT(BT != "float" || (AT == "float" && DT == "float"))
VARIANTS(
    B_PROLOGUE,
    GemmBPrologueKind::FullPrecision,
    GemmBPrologueKind::ScaleBiasDequant,
    GemmBPrologueKind::ScaleZeroPointDequant,
    GemmBPrologueKind::ScaleSymmetricDequant)
VARIANTS(GROUP_SIZE, 0, 16, 32, 64, 128)
VARIANTS(BITS, 0, 4, 8)
VARIANTS(K_SPLIT, 1, 2, 4, 8)
VARIANTS(INPUT_ALIGNED, false, true)
VARIANTS(INPUT_ROW_TILE, 1, 2, 3, 4, 5, 6, 7, 8)
VARIANTS(OUTPUT_ROW_TILE, 1, 2, 4, 8, 16, 32, 64)
VARIANTS(REDUCTION_LANES, 8, 16, 32)
VARIANTS(GROUP_LANES, 1, 2, 4, 8, 16)
VARIANTS(NUM_SIMDGROUPS, 2, 4, 8)

CONSTRAINT((B_PROLOGUE == GemmBPrologueKind::FullPrecision) == (BITS == 0))
CONSTRAINT((BITS == 0) == (GROUP_SIZE == 0))
CONSTRAINT(B_PROLOGUE == GemmBPrologueKind::FullPrecision || BT != "float")
CONSTRAINT(BITS == 0 || K_SPLIT == 1)
CONSTRAINT(BITS != 0 || (INPUT_ROW_TILE == 1 && REDUCTION_LANES == 32 && NUM_SIMDGROUPS == 8 && GROUP_LANES == 1))
CONSTRAINT(INPUT_ROW_TILE != 1 || REDUCTION_LANES == 32)
CONSTRAINT(
    INPUT_ROW_TILE == 1 ||
    (K_SPLIT == 1 && INPUT_ALIGNED && AT == "bfloat" && DT == "bfloat" && GROUP_LANES == 1))

#define GEMV_TILE(OUTPUT_ROWS, LANES, SIMD_GROUPS)                                                                     \
  (OUTPUT_ROW_TILE == OUTPUT_ROWS && REDUCTION_LANES == LANES && NUM_SIMDGROUPS == SIMD_GROUPS)
#define INPUT_ROWS(FIRST, LAST) (INPUT_ROW_TILE >= FIRST && INPUT_ROW_TILE <= LAST)

CONSTRAINT(
    INPUT_ROW_TILE == 1 ||
    (((B_PROLOGUE == GemmBPrologueKind::ScaleSymmetricDequant && BITS == 8) ||
      (B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant && BITS == 4)) &&
     (GROUP_SIZE == 32 || GROUP_SIZE == 64) &&
     ((GEMV_TILE(16, 8, 2) && INPUT_ROWS(2, 7)) ||
      (GEMV_TILE(16, 16, 4) && INPUT_ROWS(2, 6) &&
       ((GROUP_SIZE == 32 && (BITS == 4 || INPUT_ROW_TILE >= 3)) ||
        (BITS == 4 && GROUP_SIZE == 64 && INPUT_ROW_TILE >= 3))) ||
      (B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant && GEMV_TILE(8, 8, 2) &&
       ((GROUP_SIZE == 32 && (INPUT_ROW_TILE == 4 || INPUT_ROW_TILE == 8)) ||
        (GROUP_SIZE == 64 && INPUT_ROW_TILE == 2))))))

// Keep only selector-reachable geometry families.
CONSTRAINT(
    BITS == 0 ||
    (NUM_SIMDGROUPS == 2 &&
     (OUTPUT_ROW_TILE == 2 || OUTPUT_ROW_TILE == 4 || OUTPUT_ROW_TILE == 8 ||
      (INPUT_ROW_TILE > 1 && OUTPUT_ROW_TILE == 16))) ||
    (NUM_SIMDGROUPS == 4 && (OUTPUT_ROW_TILE == 8 || OUTPUT_ROW_TILE == 16 || OUTPUT_ROW_TILE == 32)) ||
    (NUM_SIMDGROUPS == 8 &&
     (OUTPUT_ROW_TILE == 16 || OUTPUT_ROW_TILE == 32 ||
      (BITS == 4 && INPUT_ALIGNED && OUTPUT_ROW_TILE == 64))))
CONSTRAINT(
    BITS == 0 || (AT == "bfloat" && DT == "bfloat") ||
    (NUM_SIMDGROUPS == 8 && OUTPUT_ROW_TILE == 32))
CONSTRAINT(BITS != 8 || (NUM_SIMDGROUPS == 8 && OUTPUT_ROW_TILE == 32) || INPUT_ROW_TILE > 1)
CONSTRAINT(INPUT_ALIGNED || K_SPLIT == 1)
CONSTRAINT(K_SPLIT <= NUM_SIMDGROUPS && NUM_SIMDGROUPS % K_SPLIT == 0)
CONSTRAINT(OUTPUT_ROW_TILE % (NUM_SIMDGROUPS / K_SPLIT) == 0)
CONSTRAINT(OUTPUT_ROW_TILE / (NUM_SIMDGROUPS / K_SPLIT) % (32 / REDUCTION_LANES) == 0)
CONSTRAINT(
    (BITS == 0 && (OUTPUT_ROW_TILE == (NUM_SIMDGROUPS / K_SPLIT) ||
                   OUTPUT_ROW_TILE == 4 * (NUM_SIMDGROUPS / K_SPLIT))) ||
    (BITS != 0 &&
     (OUTPUT_ROW_TILE == NUM_SIMDGROUPS || OUTPUT_ROW_TILE == 2 * NUM_SIMDGROUPS ||
      OUTPUT_ROW_TILE == 4 * NUM_SIMDGROUPS || OUTPUT_ROW_TILE == 8 * NUM_SIMDGROUPS)))
CONSTRAINT(REDUCTION_LANES % GROUP_LANES == 0)
CONSTRAINT(BITS == 0 || GROUP_SIZE % GROUP_LANES == 0)
CONSTRAINT(BITS != 4 || (GROUP_SIZE / GROUP_LANES) % 16 == 0)
CONSTRAINT(BITS != 8 || (GROUP_SIZE / GROUP_LANES) % 8 == 0)
CONSTRAINT(BITS != 0 || REDUCTION_LANES == 32)
CONSTRAINT(
    BITS == 0 || INPUT_ROW_TILE > 1 || (BITS == 4 &&
     ((GROUP_SIZE == 16 && GROUP_LANES == 1) || (GROUP_SIZE == 32 && GROUP_LANES == 2) ||
      (GROUP_SIZE == 64 && GROUP_LANES == 4) || (GROUP_SIZE == 128 && GROUP_LANES == 8))) ||
    (BITS == 8 &&
     ((GROUP_SIZE == 16 && GROUP_LANES == 2) || (GROUP_SIZE == 32 && GROUP_LANES == 4) ||
      (GROUP_SIZE == 64 && GROUP_LANES == 8) || (GROUP_SIZE == 128 && GROUP_LANES == 16))))
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
    const constant float& soft_cap
        OPTIONAL(output_transform.contains(GemmDTransform::SOFT_CAP)),
    const GemmDTransform output_transform SPECIALIZE,
    const bool gathered SPECIALIZE,
    const bool signed_codes SPECIALIZE,
    const bool full_tile SPECIALIZE,
    threadgroup float shared_results[INPUT_ROW_TILE * OUTPUT_ROW_TILE * K_SPLIT],
    const uint input_tile_idx GROUPS(batch_size.div_ceil(INPUT_ROW_TILE)),
    const uint output_tile_idx GROUPS(group_count_x),
    const uint simd_lane THREADS(32),
    const uint simd_group THREADS(NUM_SIMDGROUPS)
) {
  using Ops = GemvOperands<AT, BT, DT>;
  const Ops ops = {b, scales, zero_points, biases, a, d, output_bias, hadamard_factors, gather_indices};
  const GemvParams params =
      {in_vec_size, out_vec_size, batch_size, ab_scale, soft_cap, output_transform, gathered, signed_codes};
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
