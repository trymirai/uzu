#include <metal_stdlib>

#include "../common/defines.h"
#include "../common/dsl.h"
#include "../generated/gemm.h"
#include "../matmul/gemv/common/arguments.h"
#include "../matmul/gemv/common/epilogue.h"
#include "../matmul/gemv/common/quant_b_source.h"
#include "../matmul/gemv/common/reduce.h"
#include "../matmul/gemv/common/tile.h"
#include "common.h"

using namespace uzu;
using namespace uzu::gemm;

// Requires in_vec_size % (BITS == 4 ? 512 : 256) == 0, out_vec_size % MOE_GEMV_OUTPUT_ROW_TILE == 0, and every
// expert plane's byte and scale offsets within uint.
template <
    typename T,
    uint GROUP_SIZE,
    uint BITS>
VARIANTS(T, bfloat)
VARIANTS(GROUP_SIZE, 64)
VARIANTS(BITS, 4, 8)
PUBLIC KERNEL(MoeGemv)(
    const device uint8_t* b,
    const device T* scales,
    const device uint8_t* zero_points OPTIONAL(BITS == 4),
    const device T* a,
    const device uint* expert_ids,
    device T* d,
    const device int32_t* hadamard_factors,
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& num_routes,
    const constant uint& active_experts_per_token,
    const constant uint& group_count_x,
    const constant uint& b_expert_stride_bytes,
    const constant uint& scale_expert_stride,
    const constant uint& zero_point_expert_stride,
    const constant uint& scale_output_stride,
    const constant uint& scale_group_stride,
    const constant uint& zero_point_output_stride,
    const constant uint& zero_point_group_stride,
    const bool a_per_token SPECIALIZE,
    threadgroup float shared_results[MOE_GEMV_OUTPUT_ROW_TILE],
    const uint route_idx GROUPS(num_routes),
    const uint output_tile_idx GROUPS(group_count_x),
    const uint simd_lane THREADS(METAL_SIMD_SIZE),
    const uint simd_group THREADS(MOE_GEMV_NUM_SIMDGROUPS)
) {
  const uint expert_id = expert_ids[route_idx];
  const uint a_row_idx = a_per_token ? (route_idx / active_experts_per_token) : route_idx;

  const device uint32_t* expert_b = reinterpret_cast<const device uint32_t*>(b + expert_id * b_expert_stride_bytes);
  const device T* expert_scales = scales + expert_id * scale_expert_stride;
  const device uint8_t* expert_zero_points =
      (BITS == 4) ? (zero_points + expert_id * zero_point_expert_stride) : nullptr;
  const device T* route_a = a + a_row_idx * in_vec_size;
  device T* route_d = d + route_idx * out_vec_size;
  const device int32_t* expert_hadamard = hadamard_factors + expert_id * out_vec_size;

  using Ops = GemvOperands<T, T, T>;
  const Ops ops = {
      expert_b,
      expert_scales,
      expert_zero_points,
      /* biases = */ nullptr,
      route_a,
      route_d,
      /* output_bias = */ nullptr,
      expert_hadamard,
      /* gather_indices = */ nullptr
  };
  const GemvParams params = {
      in_vec_size,
      out_vec_size,
      scale_output_stride,
      scale_group_stride,
      zero_point_output_stride,
      zero_point_group_stride,
      /* batch_size = */ 1u,
      /* ab_scale = */ 1.0f,
      /* soft_cap = */ 0.0f,
      GemmDTransform(GemmDTransform::RHT),
      /* gathered = */ false,
      /* signed_codes = */ false,
      GemmBLayout::RowMajor
  };

  // A lane stages 8 bytes of codes: 16 W4 or 8 W8 values.
  constexpr uint GROUP_LANES = GROUP_SIZE * BITS / 64;
  using Tile = GemvTile<
      /* INPUT_ROW_TILE = */ 1,
      /* OUTPUT_ROW_TILE = */ MOE_GEMV_OUTPUT_ROW_TILE,
      /* REDUCTION_LANES = */ 32,
      GROUP_LANES,
      /* NUM_SIMDGROUPS = */ MOE_GEMV_NUM_SIMDGROUPS,
      /* K_SPLIT = */ 1>;

  const OutputTile<Tile, /* FULL_TILE = */ true> tile = OutputTile<Tile, /* FULL_TILE = */ true>::make(
      output_tile_idx,
      /* input_tile_idx = */ 0u,
      simd_group,
      simd_lane,
      out_vec_size
  );
  thread float result[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE] = {{0}};

  // 4-bit weights carry zero points, 8-bit weights are symmetric.
  constexpr GemmBPrologueKind B_PROLOGUE =
      BITS == 4 ? GemmBPrologueKind::ScaleZeroPointDequant : GemmBPrologueKind::ScaleSymmetricDequant;
  QuantBSource<
      Tile,
      T,
      T,
      T,
      B_PROLOGUE,
      GROUP_SIZE,
      BITS,
      /* INPUT_ALIGNED = */ true,
      /* FULL_TILE = */ true>::accumulate(result, ops, params, tile);

  Reduce<Tile, /* FULL_TILE = */ true>::run(result, shared_results, tile);
  Epilogue<Tile, T, T, T, /* FULL_TILE = */ true>::store(result, ops, params, tile, shared_results);
}
