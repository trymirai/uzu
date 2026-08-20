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
VARIANTS(INPUT_ROW_TILE, 1)
VARIANTS(OUTPUT_ROW_TILE, 1, 2, 4, 8, 16, 32, 64)
// Production uses one 32-lane reduction schedule.
VARIANTS(REDUCTION_LANES, 32)
VARIANTS(GROUP_LANES, 1, 2, 4, 8, 16)
VARIANTS(NUM_SIMDGROUPS, 2, 4, 8)
VARIANTS(PREFETCHED, false, true)
VARIANTS(FULL_TILE, false, true)

CONSTRAINT((B_PROLOGUE == GemmBPrologueKind::FullPrecision) == (BITS == 0))
CONSTRAINT((BITS == 0) == (GROUP_SIZE == 0))
CONSTRAINT(B_PROLOGUE == GemmBPrologueKind::FullPrecision || BT != "float")
CONSTRAINT(BITS == 0 || K_SPLIT == 1)
CONSTRAINT(BITS != 0 || !PREFETCHED)
CONSTRAINT(BITS != 0 || (INPUT_ROW_TILE == 1 && REDUCTION_LANES == 32 && NUM_SIMDGROUPS == 8 && GROUP_LANES == 1))
// Keep only selector-reachable geometry families.
CONSTRAINT(
    BITS == 0 ||
    (NUM_SIMDGROUPS == 2 && (OUTPUT_ROW_TILE == 2 || OUTPUT_ROW_TILE == 4 || OUTPUT_ROW_TILE == 8)) ||
    (NUM_SIMDGROUPS == 4 && (OUTPUT_ROW_TILE == 8 || OUTPUT_ROW_TILE == 16 || OUTPUT_ROW_TILE == 32)) ||
    (NUM_SIMDGROUPS == 8 &&
     (OUTPUT_ROW_TILE == 16 || OUTPUT_ROW_TILE == 32 ||
      (BITS == 4 && INPUT_ALIGNED && OUTPUT_ROW_TILE == 64))))
CONSTRAINT(
    BITS == 0 || (AT == "bfloat" && DT == "bfloat") ||
    (NUM_SIMDGROUPS == 8 && OUTPUT_ROW_TILE == 32))
CONSTRAINT(BITS != 8 || (NUM_SIMDGROUPS == 8 && OUTPUT_ROW_TILE == 32))
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
    BITS == 0 || (BITS == 4 &&
     ((GROUP_SIZE == 16 && GROUP_LANES == 1) || (GROUP_SIZE == 32 && GROUP_LANES == 2) ||
      (GROUP_SIZE == 64 && GROUP_LANES == 4) || (GROUP_SIZE == 128 && GROUP_LANES == 8))) ||
    (BITS == 8 &&
     ((GROUP_SIZE == 16 && GROUP_LANES == 2) || (GROUP_SIZE == 32 && GROUP_LANES == 4) ||
      (GROUP_SIZE == 64 && GROUP_LANES == 8) || (GROUP_SIZE == 128 && GROUP_LANES == 16))))
// Tail-free variants require exact tile divisibility.
CONSTRAINT(BITS != 0 || OUTPUT_ROW_TILE != 1 || FULL_TILE)
CONSTRAINT(BITS == 0 || OUTPUT_ROW_TILE != 64 || FULL_TILE)
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
    const bool prefetched SPECIALIZE,
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
    using Tile =
        GemvTile<INPUT_ROW_TILE, OUTPUT_ROW_TILE, REDUCTION_LANES, GROUP_LANES, NUM_SIMDGROUPS, K_SPLIT, FullTile>;
  const Tile tile = Tile::make(output_tile_idx, input_tile_idx, simd_group, simd_lane, out_vec_size);
  thread float result[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE] = {{0}};

  if constexpr (BITS == 0) {
    FullPrecisionBSource<Tile, AT, BT, DT, INPUT_ALIGNED>::accumulate(result, ops, params, tile);
  } else {
      dispatch_bool(prefetched, [&](auto prefetched_constant) {
        constexpr bool Prefetched = decltype(prefetched_constant)::value;
        QuantBSource<Tile, AT, BT, DT, B_PROLOGUE, GROUP_SIZE, BITS, INPUT_ALIGNED, Prefetched>::accumulate(
        result,
        ops,
        params,
        tile
    );
      });
  }

  Reduce<Tile>::run(result, shared_results, tile);
  Epilogue<Tile, AT, BT, DT>::store(result, ops, params, tile, shared_results);
  });
}
