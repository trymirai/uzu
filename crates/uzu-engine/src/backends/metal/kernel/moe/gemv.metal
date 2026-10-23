#include <metal_stdlib>

#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/integral_constant.h"
#include "../generated/gemm.h"
#include "../matmul/gemv/common/arguments.h"
#include "../matmul/gemv/common/epilogue.h"
#include "../matmul/gemv/common/full_precision_b_source.h"
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
VARIANTS(GROUP_SIZE, 0, 32, 64)
VARIANTS(BITS, 0, 4, 8)
CONSTRAINT((BITS == 0) == (GROUP_SIZE == 0))
PUBLIC KERNEL(MoeGemv)(
    const device uint8_t* b,
    const device T* scales OPTIONAL(b_prologue != GemmBPrologueKind::FullPrecision),
    const device uint8_t* zero_points OPTIONAL(b_prologue == GemmBPrologueKind::ScaleZeroPointDequant),
    const device T* biases OPTIONAL(b_prologue == GemmBPrologueKind::ScaleBiasDequant),
    const device uint8_t* shared_b OPTIONAL(has_shared),
    const device T* shared_scales OPTIONAL(has_shared && b_prologue != GemmBPrologueKind::FullPrecision),
    const device uint8_t* shared_zero_points OPTIONAL(has_shared && b_prologue == GemmBPrologueKind::ScaleZeroPointDequant),
    const device T* shared_biases OPTIONAL(has_shared && b_prologue == GemmBPrologueKind::ScaleBiasDequant),
    const device T* a,
    const device uint* expert_ids,
    device T* d,
    const device T* output_bias OPTIONAL(output_transform.contains(GemmDTransform::BIAS)),
    const device int32_t* hadamard_factors OPTIONAL(output_transform.contains(GemmDTransform::RHT)),
    const device T* shared_output_bias OPTIONAL(has_shared && output_transform.contains(GemmDTransform::BIAS)),
    const device int32_t* shared_hadamard_factors OPTIONAL(has_shared && output_transform.contains(GemmDTransform::RHT)),
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
    const constant uint& num_routed_routes,
    const GemmBPrologueKind b_prologue SPECIALIZE,
    const GemmDTransform output_transform SPECIALIZE,
    const bool input_aligned SPECIALIZE,
    const bool signed_codes SPECIALIZE,
    const bool full_tile SPECIALIZE,
    const bool a_per_token SPECIALIZE,
    const bool has_shared SPECIALIZE,
    threadgroup float shared_results[METAL_SIMD_SIZE],
    const uint route_idx GROUPS(num_routes),
    const uint output_tile_idx GROUPS(group_count_x),
    const uint simd_lane THREADS(METAL_SIMD_SIZE),
    const uint simd_group THREADS(MOE_GEMV_NUM_SIMDGROUPS)
) {
  const bool is_shared = has_shared && (route_idx >= num_routed_routes);
  const uint expert_id = is_shared ? 0u : expert_ids[route_idx];
  const uint a_row_idx =
      a_per_token ? (is_shared ? (route_idx - num_routed_routes) : (route_idx / active_experts_per_token)) : route_idx;

  // Offsets are 32-bit: the MoE block checks at load that every expert plane fits in uint.
  const device uint32_t* expert_b =
      is_shared ? reinterpret_cast<const device uint32_t*>(shared_b)
                : reinterpret_cast<const device uint32_t*>(b + expert_id * b_expert_stride_bytes);
  const device T* expert_scales = (b_prologue != GemmBPrologueKind::FullPrecision)
                                      ? (is_shared ? shared_scales : (scales + expert_id * scale_expert_stride))
                                      : nullptr;
  const device uint8_t* expert_zero_points =
      (b_prologue == GemmBPrologueKind::ScaleZeroPointDequant)
          ? (is_shared ? shared_zero_points : (zero_points + expert_id * zero_point_expert_stride))
          : nullptr;
  const device T* expert_biases = (b_prologue == GemmBPrologueKind::ScaleBiasDequant)
                                      ? (is_shared ? shared_biases : (biases + expert_id * scale_expert_stride))
                                      : nullptr;
  const device T* route_a = a + a_row_idx * in_vec_size;
  device T* route_d = d + route_idx * out_vec_size;
  const device T* expert_output_bias = output_transform.contains(GemmDTransform::BIAS)
                                           ? (is_shared ? shared_output_bias : (output_bias + expert_id * out_vec_size))
                                           : nullptr;
  const device int32_t* expert_hadamard =
      output_transform.contains(GemmDTransform::RHT)
          ? (is_shared ? shared_hadamard_factors : (hadamard_factors + expert_id * out_vec_size))
          : nullptr;

  using Ops = GemvOperands<T, T, T>;
  const Ops ops = {
      expert_b,
      expert_scales,
      expert_zero_points,
      expert_biases,
      route_a,
      route_d,
      expert_output_bias,
      expert_hadamard,
      nullptr
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
      output_transform,
      /* gathered = */ false,
      signed_codes,
      GemmBLayout::RowMajor
  };

  constexpr uint GROUP_LANES =
      (BITS == 0) ? 1 : ((BITS == 4) ? (GROUP_SIZE == 32 ? 2 : 4) : (GROUP_SIZE == 32 ? 4 : 8));
  using Tile = GemvTile<
      /* INPUT_ROW_TILE = */ 1,
      /* OUTPUT_ROW_TILE = */ 32,
      /* REDUCTION_LANES = */ 32,
      GROUP_LANES,
      /* NUM_SIMDGROUPS = */ MOE_GEMV_NUM_SIMDGROUPS,
      /* K_SPLIT = */ 1>;

  dispatch_bool(full_tile, [&](auto full_tile_constant) {
    constexpr bool FullTile = decltype(full_tile_constant)::value;
    dispatch_bool(input_aligned, [&](auto input_aligned_constant) {
      constexpr bool InputAligned = decltype(input_aligned_constant)::value;
      const OutputTile<Tile, FullTile> tile = OutputTile<Tile, FullTile>::make(
          output_tile_idx,
          /* input_tile_idx = */ 0u,
          simd_group,
          simd_lane,
          out_vec_size
      );
      thread float result[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE] = {{0}};

      if constexpr (BITS == 0) {
        FullPrecisionBSource<Tile, T, T, T, InputAligned, FullTile>::accumulate(result, ops, params, tile);
      } else if constexpr (BITS == 4) {
        if (b_prologue == GemmBPrologueKind::ScaleZeroPointDequant) {
          QuantBSource<
              Tile,
              T,
              T,
              T,
              GemmBPrologueKind::ScaleZeroPointDequant,
              GROUP_SIZE,
              BITS,
              InputAligned,
              FullTile>::accumulate(result, ops, params, tile);
        } else {
          QuantBSource<Tile, T, T, T, GemmBPrologueKind::ScaleBiasDequant, GROUP_SIZE, BITS, InputAligned, FullTile>::
              accumulate(result, ops, params, tile);
        }
      } else {
        QuantBSource<
            Tile,
            T,
            T,
            T,
            GemmBPrologueKind::ScaleSymmetricDequant,
            GROUP_SIZE,
            BITS,
            InputAligned,
            FullTile>::accumulate(result, ops, params, tile);
      }

      Reduce<Tile, FullTile>::run(result, shared_results, tile);
      Epilogue<Tile, T, T, T, FullTile>::store(result, ops, params, tile, shared_results);
    });
  });
}
