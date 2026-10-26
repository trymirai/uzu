#include <metal_stdlib>

#include "../attention/attention_gate.h"
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
PUBLIC KERNEL(MoeDownCombine)(
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
    const device float* expert_weights,
    const device float* shared_gate OPTIONAL(has_gate),
    device T* output,
    const device T* output_bias OPTIONAL(output_transform.contains(GemmDTransform::BIAS)),
    const device int32_t* hadamard_factors OPTIONAL(output_transform.contains(GemmDTransform::RHT)),
    const device T* shared_output_bias OPTIONAL(has_shared && output_transform.contains(GemmDTransform::BIAS)),
    const device int32_t* shared_hadamard_factors OPTIONAL(has_shared && output_transform.contains(GemmDTransform::RHT)),
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& active_experts_per_token,
    const constant uint& group_count_x,
    const constant uint& b_expert_stride_bytes,
    const constant uint& scale_expert_stride,
    const constant uint& zero_point_expert_stride,
    const constant uint& scale_output_stride,
    const constant uint& scale_group_stride,
    const constant uint& zero_point_output_stride,
    const constant uint& zero_point_group_stride,
    const constant uint& batch_size,
    const constant uint& gate_stride,
    const GemmBPrologueKind b_prologue SPECIALIZE,
    const GemmDTransform output_transform SPECIALIZE,
    const bool input_aligned SPECIALIZE,
    const bool signed_codes SPECIALIZE,
    const bool full_tile SPECIALIZE,
    const bool has_shared SPECIALIZE,
    const bool has_gate SPECIALIZE,
    threadgroup float shared_results[METAL_SIMD_SIZE],
    const uint output_tile_idx GROUPS(group_count_x),
    const uint token_idx GROUPS(batch_size),
    const uint simd_lane THREADS(METAL_SIMD_SIZE),
    const uint simd_group THREADS(MOE_GEMV_NUM_SIMDGROUPS)
) {
  constexpr uint GROUP_LANES =
      (BITS == 0) ? 1 : ((BITS == 4) ? (GROUP_SIZE == 32 ? 2 : 4) : (GROUP_SIZE == 32 ? 4 : 8));
  using Tile = GemvTile<
      /* INPUT_ROW_TILE = */ 1,
      /* OUTPUT_ROW_TILE = */ MOE_GEMV_OUTPUT_ROW_TILE,
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
      thread float combined[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE] = {{0}};

      const uint base_routed_idx = token_idx * active_experts_per_token;
      const device float* weights_row = expert_weights + base_routed_idx;

      for (uint k = 0; k < active_experts_per_token; ++k) {
        const float weight = weights_row[k];
        const uint route_idx = base_routed_idx + k;
        const uint expert_id = expert_ids[route_idx];

        const device uint32_t* expert_b =
            reinterpret_cast<const device uint32_t*>(b + expert_id * b_expert_stride_bytes);
        const device T* expert_scales =
            (b_prologue != GemmBPrologueKind::FullPrecision) ? (scales + expert_id * scale_expert_stride) : nullptr;
        const device uint8_t* expert_zero_points = (b_prologue == GemmBPrologueKind::ScaleZeroPointDequant)
                                                       ? (zero_points + expert_id * zero_point_expert_stride)
                                                       : nullptr;
        const device T* expert_biases =
            (b_prologue == GemmBPrologueKind::ScaleBiasDequant) ? (biases + expert_id * scale_expert_stride) : nullptr;
        const device T* route_a = a + route_idx * in_vec_size;

        using Ops = GemvOperands<T, T, T>;
        const Ops ops =
            {expert_b, expert_scales, expert_zero_points, expert_biases, route_a, nullptr, nullptr, nullptr, nullptr};
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
            GemmDTransform(),
            /* gathered = */ false,
            signed_codes,
            GemmBLayout::RowMajor
        };

        thread float route_res[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE] = {{0}};

        if constexpr (BITS == 0) {
          FullPrecisionBSource<Tile, T, T, T, InputAligned, FullTile>::accumulate(route_res, ops, params, tile);
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
                FullTile>::accumulate(route_res, ops, params, tile);
          } else {
            QuantBSource<Tile, T, T, T, GemmBPrologueKind::ScaleBiasDequant, GROUP_SIZE, BITS, InputAligned, FullTile>::
                accumulate(route_res, ops, params, tile);
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
              FullTile>::accumulate(route_res, ops, params, tile);
        }

        Reduce<Tile, FullTile>::run(route_res, nullptr, tile);

        const device T* expert_output_bias =
            output_transform.contains(GemmDTransform::BIAS) ? (output_bias + expert_id * out_vec_size) : nullptr;

        Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
          constexpr uint R = decltype(output_index)::value;
          const uint global_row = tile.row0 + R;
          if (FullTile || global_row < out_vec_size) {
            float val = route_res[0][R];
            if (output_transform.contains(GemmDTransform::BIAS)) {
              val += static_cast<float>(expert_output_bias[global_row]);
            }
            combined[0][R] = fma(weight, val, combined[0][R]);
          }
        });
      }

      if (has_shared) {
        const uint num_routed_routes = batch_size * active_experts_per_token;
        const uint shared_route_idx = num_routed_routes + token_idx;
        const device uint32_t* expert_b = reinterpret_cast<const device uint32_t*>(shared_b);
        const device T* expert_scales = (b_prologue != GemmBPrologueKind::FullPrecision) ? shared_scales : nullptr;
        const device uint8_t* expert_zero_points =
            (b_prologue == GemmBPrologueKind::ScaleZeroPointDequant) ? shared_zero_points : nullptr;
        const device T* expert_biases = (b_prologue == GemmBPrologueKind::ScaleBiasDequant) ? shared_biases : nullptr;
        const device T* route_a = a + shared_route_idx * in_vec_size;

        using Ops = GemvOperands<T, T, T>;
        const Ops ops =
            {expert_b, expert_scales, expert_zero_points, expert_biases, route_a, nullptr, nullptr, nullptr, nullptr};
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
            GemmDTransform(),
            /* gathered = */ false,
            signed_codes,
            GemmBLayout::RowMajor
        };

        thread float shared_res[Tile::INPUT_ROWS][Tile::ROWS_PER_LANE] = {{0}};

        if constexpr (BITS == 0) {
          FullPrecisionBSource<Tile, T, T, T, InputAligned, FullTile>::accumulate(shared_res, ops, params, tile);
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
                FullTile>::accumulate(shared_res, ops, params, tile);
          } else {
            QuantBSource<Tile, T, T, T, GemmBPrologueKind::ScaleBiasDequant, GROUP_SIZE, BITS, InputAligned, FullTile>::
                accumulate(shared_res, ops, params, tile);
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
              FullTile>::accumulate(shared_res, ops, params, tile);
        }

        Reduce<Tile, FullTile>::run(shared_res, nullptr, tile);

        const device T* expert_output_bias =
            output_transform.contains(GemmDTransform::BIAS) ? (shared_output_bias + 0 * out_vec_size) : nullptr;

        float shared_gate_val = 0.0f;
        if (has_gate) {
          shared_gate_val = shared_gate[token_idx * gate_stride];
        }

        Tile::for_each_output_row([&](auto output_index) UZU_ALWAYS_INLINE {
          constexpr uint R = decltype(output_index)::value;
          const uint global_row = tile.row0 + R;
          if (FullTile || global_row < out_vec_size) {
            float val = shared_res[0][R];
            if (output_transform.contains(GemmDTransform::BIAS)) {
              val += static_cast<float>(expert_output_bias[global_row]);
            }
            if (has_gate) {
              val = attention_apply_gate(val, shared_gate_val);
            }
            combined[0][R] += val;
          }
        });
      }

      const device int32_t* token_hadamard =
          output_transform.contains(GemmDTransform::RHT) ? hadamard_factors : nullptr;
      using Ops = GemvOperands<T, T, T>;
      const Ops ops_out = {
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          output + token_idx * out_vec_size,
          nullptr,
          token_hadamard,
          nullptr
      };
      const GemvParams params_out = {
          in_vec_size,
          out_vec_size,
          scale_output_stride,
          scale_group_stride,
          zero_point_output_stride,
          zero_point_group_stride,
          /* batch_size = */ 1u,
          /* ab_scale = */ 1.0f,
          /* soft_cap = */ 0.0f,
          GemmDTransform(output_transform.bits() & ~GemmDTransform::BIAS),
          /* gathered = */ false,
          signed_codes,
          GemmBLayout::RowMajor
      };

      Epilogue<Tile, T, T, T, FullTile>::store(combined, ops_out, params_out, tile, shared_results);
    });
  });
}
