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
    uint GROUP_SIZE,
    uint BITS,
    uint NUM_SIMDGROUPS>
VARIANTS(AT, bfloat, float)
VARIANTS(BT, bfloat, float)
VARIANTS(DT, bfloat, float)
CONSTRAINT(BT != "float")
VARIANTS(GROUP_SIZE, 16, 32, 64, 128)
VARIANTS(BITS, 4)
VARIANTS(NUM_SIMDGROUPS, 8)
KERNEL(GemvLloydMax)(
    const device uint32_t* b,
    const device BT* scales,
    const device half* codebook,
    const device uint8_t* bias_indices,
    const device half* bias_codebook,
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
    threadgroup float shared_results[NUM_SIMDGROUPS * 4],
    const uint batch_idx GROUPS(batch_size),
    const uint out_block_idx GROUPS(group_count_x),
    const uint simd_lane THREADS(32),
    const uint simd_group THREADS(NUM_SIMDGROUPS)
) {
  typedef float U;
  thread U result[RESULTS_PER_SIMDGROUP] = {0};

  OutputTile<1, NUM_SIMDGROUPS> tile =
      OutputTile<1, NUM_SIMDGROUPS>::make(
          out_block_idx,
          simd_group,
          out_vec_size
      );
  d += batch_idx * out_vec_size + tile.out_row;

  LloydMaxBSource<BT, AT, U, GROUP_SIZE, BITS>::accumulate(
      result,
      b,
      scales,
      codebook,
      bias_indices,
      bias_codebook,
      a,
      in_vec_size,
      tile.out_row,
      batch_idx,
      simd_lane
  );

  Reduce<U, 1, NUM_SIMDGROUPS>::run(
      result,
      shared_results,
      simd_group,
      simd_lane,
      tile.row_group,
      tile.k_slice
  );

  Epilogue<BT, DT, U>::store(
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
