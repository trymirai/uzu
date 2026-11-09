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
    threadgroup float shared_results[moe::GEMV_OUTPUT_ROW_TILE],
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

  const device T* biases = nullptr;
  const device T* output_bias = nullptr;
  const device uint* gather_indices = nullptr;
  using Ops = GemvOperands<T, T, T>;
  const Ops ops = {
      expert_b,
      expert_scales,
      expert_zero_points,
      biases,
      route_a,
      route_d,
      output_bias,
      expert_hadamard,
      gather_indices
  };
  constexpr uint batch_size = 1;
  constexpr float ab_scale = 1.0f;
  constexpr float soft_cap = 0.0f;
  constexpr bool gathered = false;
  constexpr bool signed_codes = false;
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
      GemmDTransform(GemmDTransform::RHT),
      gathered,
      signed_codes,
      GemmBLayout::RowMajor
  };

  constexpr uint INPUT_ROW_TILE = 1;
  constexpr uint REDUCTION_LANES = 32;
  // A lane stages 8 bytes of codes: 16 W4 or 8 W8 values.
  constexpr uint GROUP_LANES = GROUP_SIZE * BITS / 64;
  constexpr uint K_SPLIT = 1;
  constexpr bool INPUT_ALIGNED = true;
  constexpr bool FULL_TILE = true;
  using Tile = GemvTile<
      INPUT_ROW_TILE,
      moe::GEMV_OUTPUT_ROW_TILE,
      REDUCTION_LANES,
      GROUP_LANES,
      MOE_GEMV_NUM_SIMDGROUPS,
      K_SPLIT>;

  constexpr uint input_tile_idx = 0;
  const OutputTile<Tile, FULL_TILE> tile =
      OutputTile<Tile, FULL_TILE>::make(output_tile_idx, input_tile_idx, simd_group, simd_lane, out_vec_size);
  thread float result[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE] = {{0}};

  constexpr GemmBPrologueKind B_PROLOGUE =
      BITS == 4 ? GemmBPrologueKind::ScaleZeroPointDequant : GemmBPrologueKind::ScaleSymmetricDequant;
  QuantBSource<Tile, T, T, T, B_PROLOGUE, GROUP_SIZE, BITS, INPUT_ALIGNED, FULL_TILE>::accumulate(
      result,
      ops,
      params,
      tile
  );

  Reduce<Tile, FULL_TILE>::run(result, shared_results, tile);
  Epilogue<Tile, T, T, T, FULL_TILE>::store(result, ops, params, tile, shared_results);
}
