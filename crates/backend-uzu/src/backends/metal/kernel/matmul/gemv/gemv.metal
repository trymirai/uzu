#include "../../common/dsl.h"
#include "../../generated/gemm.h"
#include "common/b_source.h"
#include "common/epilogue.h"
#include "common/output_tile.h"
#include "common/reduce.h"

using namespace metal;
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
    uint RESULTS_PER_SIMDGROUP,
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
VARIANTS(RESULTS_PER_SIMDGROUP, 4)
VARIANTS(NUM_SIMDGROUPS, 2, 8)
CONSTRAINT((B_PROLOGUE == GemmBPrologueKind::FullPrecision) == (BITS == 0))
CONSTRAINT((BITS == 0) == (GROUP_SIZE == 0))
CONSTRAINT(B_PROLOGUE == GemmBPrologueKind::FullPrecision || BT != "float")
CONSTRAINT(B_PROLOGUE == GemmBPrologueKind::FullPrecision || K_SPLIT == 1)
CONSTRAINT(B_PROLOGUE != GemmBPrologueKind::FullPrecision || NUM_SIMDGROUPS == 8)
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
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& batch_size,
    const constant float& ab_scale,
    const constant uint& group_count_x,
    const GemmDTransform output_transform SPECIALIZE,
    threadgroup float shared_results[NUM_SIMDGROUPS * RESULTS_PER_SIMDGROUP],
    const uint batch_idx GROUPS(batch_size),
    const uint out_block_idx GROUPS(group_count_x),
    const uint simd_lane THREADS(32),
    const uint simd_group THREADS(NUM_SIMDGROUPS)
) {
  typedef float U;
  thread U result[RESULTS_PER_SIMDGROUP] = {0};

  OutputTile<K_SPLIT, NUM_SIMDGROUPS, RESULTS_PER_SIMDGROUP> tile =
      OutputTile<K_SPLIT, NUM_SIMDGROUPS, RESULTS_PER_SIMDGROUP>::make(out_block_idx, simd_group, out_vec_size);
  d += batch_idx * out_vec_size + tile.out_row;

  BSource<BT, AT, U, B_PROLOGUE, GROUP_SIZE, BITS, K_SPLIT, RESULTS_PER_SIMDGROUP, INPUT_ALIGNED>::accumulate(
      result,
      b,
      scales,
      zero_points,
      biases,
      a,
      in_vec_size,
      tile.out_row,
      batch_idx,
      simd_lane,
      tile.k_slice
  );

  Reduce<U, K_SPLIT, NUM_SIMDGROUPS, RESULTS_PER_SIMDGROUP>::run(
      result,
      shared_results,
      simd_group,
      simd_lane,
      tile.row_group,
      tile.k_slice
  );

  Epilogue<BT, DT, U, RESULTS_PER_SIMDGROUP>::store(
      result,
      d,
      output_bias,
      hadamard_factors,
      shared_results,
      ab_scale,
      output_transform,
      tile.out_row,
      out_vec_size,
      out_block_idx,
      simd_group,
      simd_lane,
      tile.writer
  );
}
