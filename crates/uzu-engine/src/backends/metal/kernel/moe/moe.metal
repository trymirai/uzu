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

using namespace uzu;
using namespace uzu::gemm;

#define MOE_ROUTER_MAX_PER_LANE 16
#define MOE_GEMV_NUM_SIMDGROUPS 8

template <typename T, uint MAX_TOP_K>
VARIANTS(T, float)
VARIANTS(MAX_TOP_K, 8)
PUBLIC KERNEL(MoeRouterTopK)(
    const device T* logits,
    const device T* biases OPTIONAL(has_biases),
    device uint* expert_ids,
    device float* expert_weights,
    const constant uint& num_experts,
    const constant uint& top_k,
    const constant uint& batch_size,
    const bool has_biases SPECIALIZE,
    const uint token_idx GROUPS(batch_size),
    const uint lane THREADS(METAL_SIMD_SIZE)
) {
  const device T* row = logits + token_idx * num_experts;
  float vals[MOE_ROUTER_MAX_PER_LANE];
  for (uint i = 0; i < MOE_ROUTER_MAX_PER_LANE; ++i) {
    const uint e = lane + i * METAL_SIMD_SIZE;
    if (e < num_experts) {
      float v = float(row[e]);
      if (has_biases) {
        v += float(biases[e]);
      }
      vals[i] = isnan(v) ? -INFINITY : v;
    } else {
      vals[i] = -INFINITY;
    }
  }

  float chosen_vals[MAX_TOP_K];
  uint chosen_ids[MAX_TOP_K];
  float max_chosen = -INFINITY;

  for (uint k = 0; k < top_k; ++k) {
    float best_val = -INFINITY;
    uint best_idx = 0xFFFFFFFFu;
    for (uint i = 0; i < MOE_ROUTER_MAX_PER_LANE; ++i) {
      const uint e = lane + i * METAL_SIMD_SIZE;
      if (e < num_experts) {
        const float v = vals[i];
        if (v > best_val || (v == best_val && e < best_idx)) {
          best_val = v;
          best_idx = e;
        }
      }
    }

    for (ushort offset = METAL_SIMD_SIZE / 2; offset > 0; offset >>= 1) {
      const float other_val = simd_shuffle_xor(best_val, offset);
      const uint other_idx = simd_shuffle_xor(best_idx, offset);
      if (other_val > best_val || (other_val == best_val && other_idx < best_idx)) {
        best_val = other_val;
        best_idx = other_idx;
      }
    }

    if (best_idx >= num_experts) {
      best_idx = min(k, num_experts - 1);
    }

    if ((best_idx & (METAL_SIMD_SIZE - 1)) == lane) {
      const uint slot = best_idx / METAL_SIMD_SIZE;
      if (slot < MOE_ROUTER_MAX_PER_LANE) {
        vals[slot] = -INFINITY;
      }
    }

    chosen_vals[k] = best_val;
    chosen_ids[k] = best_idx;
    max_chosen = (k == 0) ? best_val : max(max_chosen, best_val);
  }

  if (lane == 0) {
    float sum_exp = 0.0f;
    for (uint k = 0; k < top_k; ++k) {
      const float e_val = (max_chosen == -INFINITY) ? (1.0f / float(top_k)) : fast::exp(chosen_vals[k] - max_chosen);
      chosen_vals[k] = e_val;
      sum_exp += e_val;
    }
    const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    device uint* out_ids = expert_ids + token_idx * top_k;
    device float* out_weights = expert_weights + token_idx * top_k;
    for (uint k = 0; k < top_k; ++k) {
      out_ids[k] = (chosen_ids[k] < num_experts) ? chosen_ids[k] : min(k, num_experts - 1);
      out_weights[k] = chosen_vals[k] * inv_sum;
    }
  }
}

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
    const device T* a,
    const device uint* expert_ids,
    device T* d,
    const device T* output_bias OPTIONAL(output_transform.contains(GemmDTransform::BIAS)),
    const device int32_t* hadamard_factors OPTIONAL(output_transform.contains(GemmDTransform::RHT)),
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
    const GemmBPrologueKind b_prologue SPECIALIZE,
    const GemmDTransform output_transform SPECIALIZE,
    const bool input_aligned SPECIALIZE,
    const bool signed_codes SPECIALIZE,
    const bool full_tile SPECIALIZE,
    const bool a_per_token SPECIALIZE,
    threadgroup float shared_results[METAL_SIMD_SIZE],
    const uint route_idx GROUPS(num_routes),
    const uint output_tile_idx GROUPS(group_count_x),
    const uint simd_lane THREADS(METAL_SIMD_SIZE),
    const uint simd_group THREADS(MOE_GEMV_NUM_SIMDGROUPS)
) {
  const uint expert_id = expert_ids[route_idx];
  const uint a_row_idx = a_per_token ? (route_idx / active_experts_per_token) : route_idx;

  const device uint32_t* expert_b =
      reinterpret_cast<const device uint32_t*>(b + ulong(expert_id) * b_expert_stride_bytes);
  const device T* expert_scales =
      (b_prologue != GemmBPrologueKind::FullPrecision) ? (scales + ulong(expert_id) * scale_expert_stride) : nullptr;
  const device uint8_t* expert_zero_points = (b_prologue == GemmBPrologueKind::ScaleZeroPointDequant)
                                                 ? (zero_points + ulong(expert_id) * zero_point_expert_stride)
                                                 : nullptr;
  const device T* expert_biases =
      (b_prologue == GemmBPrologueKind::ScaleBiasDequant) ? (biases + ulong(expert_id) * scale_expert_stride) : nullptr;
  const device T* route_a = a + ulong(a_row_idx) * in_vec_size;
  device T* route_d = d + ulong(route_idx) * out_vec_size;
  const device T* expert_output_bias =
      output_transform.contains(GemmDTransform::BIAS) ? (output_bias + ulong(expert_id) * out_vec_size) : nullptr;
  const device int32_t* expert_hadamard =
      output_transform.contains(GemmDTransform::RHT) ? (hadamard_factors + ulong(expert_id) * out_vec_size) : nullptr;

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

template <typename T>
VARIANTS(T, bfloat)
PUBLIC KERNEL(MoeCombine)(
    const device T* expert_outputs,
    const device float* expert_weights,
    const device T* shared_output OPTIONAL(has_shared),
    const device T* shared_gate OPTIONAL(has_gate),
    device T* output,
    const constant uint& model_dim,
    const constant uint& top_k,
    const constant uint& batch_size,
    const bool has_shared SPECIALIZE,
    const bool has_gate SPECIALIZE,
    const uint col AXIS(model_dim, 256),
    const uint token_idx AXIS(batch_size, 1)
) {
  float acc = 0.0f;
  const device float* weights_row = expert_weights + token_idx * top_k;
  const uint base_route = token_idx * top_k;
  for (uint k = 0; k < top_k; ++k) {
    const float w = weights_row[k];
    const float val = float(expert_outputs[(base_route + k) * model_dim + col]);
    acc = fma(w, val, acc);
  }
  if (has_shared) {
    float s_val = float(shared_output[token_idx * model_dim + col]);
    if (has_gate) {
      s_val = attention_apply_gate(s_val, float(shared_gate[token_idx]));
    }
    acc += s_val;
  }
  output[token_idx * model_dim + col] = T(acc);
}
