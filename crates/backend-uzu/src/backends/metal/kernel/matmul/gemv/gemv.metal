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
    uint NUM_SIMDGROUPS,
    bool MICROFLOAT>
VARIANTS(AT, half, bfloat, float)
VARIANTS(BT, half, bfloat, float)
VARIANTS(DT, half, bfloat, float)
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
VARIANTS(RESULTS_PER_SIMDGROUP, 1, 2, 4, 8)
VARIANTS(NUM_SIMDGROUPS, 2, 4, 8)
VARIANTS(MICROFLOAT, false, true)
CONSTRAINT(MICROFLOAT || (B_PROLOGUE == GemmBPrologueKind::FullPrecision) == (BITS == 0))
CONSTRAINT(MICROFLOAT || (BITS == 0) == (GROUP_SIZE == 0))
CONSTRAINT(!MICROFLOAT || (B_PROLOGUE == GemmBPrologueKind::FullPrecision && BITS == 4))
CONSTRAINT(!MICROFLOAT || GROUP_SIZE == 16 || GROUP_SIZE == 32)
CONSTRAINT(B_PROLOGUE == GemmBPrologueKind::FullPrecision || BT != "float")
CONSTRAINT(B_PROLOGUE == GemmBPrologueKind::FullPrecision || K_SPLIT == 1)
CONSTRAINT(K_SPLIT <= NUM_SIMDGROUPS)
// Only selector-reachable tiles are instantiated (fleet-tuned tables): fp
// always runs 8 simdgroups with 1 or 4 rows each; non-default quantized
// tiles exist for bf16 IO only. Widen locally when sweeping new configs.
CONSTRAINT(BITS != 0 || NUM_SIMDGROUPS == 8)
CONSTRAINT(BITS != 0 || RESULTS_PER_SIMDGROUP == 1 || RESULTS_PER_SIMDGROUP == 4)
CONSTRAINT(
    BITS == 0 || (NUM_SIMDGROUPS == 8 && RESULTS_PER_SIMDGROUP == 4) ||
    (AT == "bfloat" && DT == "bfloat"))
KERNEL(Gemv)(
    const device uint32_t* b,
    const device BT* scales
        OPTIONAL(B_PROLOGUE != GemmBPrologueKind::FullPrecision || MICROFLOAT),
    const device uint8_t* zero_points
        OPTIONAL(B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant),
    const device BT* biases
        OPTIONAL(B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant),
    const device BT* global_scales OPTIONAL(MICROFLOAT),
    const device AT* a,
    device DT* d,
    const device BT* output_bias
        OPTIONAL(output_transform.contains(GemmDTransform::BIAS)),
    const device int32_t* hadamard_factors
        OPTIONAL(output_transform.contains(GemmDTransform::RHT)),
    const device uint* gather_indices OPTIONAL(gathered),
    const device int* expert_ids OPTIONAL(expert_routed),
    const device BT* expert_biases OPTIONAL(expert_bias),
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& batch_size,
    const constant float& ab_scale,
    const constant uint& group_count_x,
    const constant uint& routes_per_token,
    const constant uint& expert_count,
    const constant bool& route_inputs,
    const constant float& soft_cap
        OPTIONAL(output_transform.contains(GemmDTransform::SOFT_CAP)),
    const GemmDTransform output_transform SPECIALIZE,
    const bool gathered SPECIALIZE,
    const bool expert_routed SPECIALIZE,
    const bool expert_bias SPECIALIZE,
    const bool signed_codes SPECIALIZE,
    threadgroup float shared_results[NUM_SIMDGROUPS * RESULTS_PER_SIMDGROUP],
    const uint batch_idx GROUPS(batch_size),
    const uint out_block_idx GROUPS(group_count_x),
    const uint simd_lane THREADS(32),
    const uint simd_group THREADS(NUM_SIMDGROUPS)
) {
  typedef float U;
  thread U result[RESULTS_PER_SIMDGROUP] = {0};

  uint matrix_idx = 0;
  uint a_row = batch_idx;
  if (expert_routed) {
    const int expert = expert_ids[batch_idx];
    if (expert < 0 || uint(expert) >= expert_count) {
      const uint thread_index = simd_group * 32 + simd_lane;
      for (uint column = thread_index; column < out_vec_size; column += NUM_SIMDGROUPS * 32) {
        d[batch_idx * out_vec_size + column] = DT(0);
      }
      return;
    }
    matrix_idx = uint(expert);
    a_row = route_inputs ? batch_idx : batch_idx / routes_per_token;
  }

  OutputTile<K_SPLIT, NUM_SIMDGROUPS, RESULTS_PER_SIMDGROUP> tile =
      OutputTile<K_SPLIT, NUM_SIMDGROUPS, RESULTS_PER_SIMDGROUP>::make(out_block_idx, simd_group, out_vec_size);
  d += batch_idx * out_vec_size + tile.out_row;

  BSource<BT, AT, U, B_PROLOGUE, GROUP_SIZE, BITS, K_SPLIT, RESULTS_PER_SIMDGROUP, INPUT_ALIGNED, MICROFLOAT>::accumulate(
      result,
      b,
      scales,
      zero_points,
      biases,
      global_scales,
      a,
      gather_indices,
      gathered,
      matrix_idx,
      in_vec_size,
      out_vec_size,
      tile.out_row,
      batch_idx,
      a_row,
      simd_lane,
      tile.k_slice,
      signed_codes
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
      expert_bias ? expert_biases + matrix_idx * out_vec_size : nullptr,
      hadamard_factors,
      shared_results,
      ab_scale,
      soft_cap,
      output_transform,
      tile.out_row,
      out_vec_size,
      out_block_idx,
      simd_group,
      simd_lane,
      tile.writer
  );
}
