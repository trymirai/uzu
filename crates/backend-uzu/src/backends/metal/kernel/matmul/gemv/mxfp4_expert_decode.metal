#include "../../common/dsl.h"
#include "../../activation/activations.h"
#include "../common/microfloat.h"

using namespace metal;

// Decode-specialized, route-direct, block-native MXFP4 expert GEMV.
//
// Selected only for small expert-routed MXFP4 projections (see
// Mxfp4ExpertDecodeGemvSpec::select in gemv/mxfp4_expert_decode.rs). Geometry
// follows the proven llama.cpp mul_mv layout: two lanes own one physical
// 32-value MXFP4 block, E2M1 decode goes through a threadgroup LUT, block
// scales are applied once per scale group, and activations are vector-loaded
// once per block and reused across the rows owned by a simdgroup.
template <
    typename AT,
    typename BT,
    typename DT,
    uint GROUP_SIZE,
    uint ROWS_PER_SIMDGROUP,
    uint NUM_SIMDGROUPS,
    bool FUSED_GATE_UP>
VARIANTS(AT, bfloat, float)
VARIANTS(BT, bfloat, float)
VARIANTS(DT, float, bfloat)
VARIANTS(GROUP_SIZE, 16, 32)
VARIANTS(ROWS_PER_SIMDGROUP, 1, 2, 4)
VARIANTS(NUM_SIMDGROUPS, 2, 4)
VARIANTS(FUSED_GATE_UP, false, true)
// The fused epilogue computes value/gate row pairs and writes SiLU-activated
// F32 hidden rows; it requires row pairs and the production W13 dtypes.
CONSTRAINT(!FUSED_GATE_UP || (AT == "bfloat" && DT == "float"))
CONSTRAINT(!FUSED_GATE_UP || ROWS_PER_SIMDGROUP != 1)
KERNEL(Mxfp4ExpertDecodeGemv)(
    const device uint8_t* codes,
    const device uint8_t* scales,
    const device BT* outer_scales,
    const device AT* a,
    device DT* d,
    const device BT* expert_biases OPTIONAL(expert_bias),
    const device int* expert_ids,
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& route_count,
    const constant uint& routes_per_token,
    const constant uint& expert_count,
    const constant bool& input_is_route_major,
    const constant uint& group_count_x,
    const constant float& activation_alpha OPTIONAL(custom_activation_alpha),
    const constant float& gate_clip_min OPTIONAL(clip_gate),
    const constant float& gate_clip_max OPTIONAL(clip_gate),
    const constant float& value_clip_min OPTIONAL(clip_value),
    const constant float& value_clip_max OPTIONAL(clip_value),
    const bool expert_bias SPECIALIZE,
    const bool custom_activation_alpha SPECIALIZE,
    const bool clip_gate SPECIALIZE,
    const bool clip_value SPECIALIZE,
    threadgroup float e2m1_lut[16],
    const uint route_idx GROUPS(route_count),
    const uint out_block_idx GROUPS(group_count_x),
    const uint simd_lane THREADS(32),
    const uint simd_group THREADS(NUM_SIMDGROUPS)
) {
  if (simd_group == 0 && simd_lane < 16) {
    e2m1_lut[simd_lane] = uzu::gemm::decode_e2m1(simd_lane);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // FUSED_GATE_UP computes value/gate row pairs per hidden output, so each
  // simdgroup owns half as many outputs and the output row width halves.
  constexpr uint OUTPUTS_PER_SIMDGROUP = FUSED_GATE_UP ? ROWS_PER_SIMDGROUP / 2 : ROWS_PER_SIMDGROUP;
  const uint hidden_dim = FUSED_GATE_UP ? out_vec_size / 2 : out_vec_size;
  const uint output_width = hidden_dim;
  const uint out_row_base =
      out_block_idx * (NUM_SIMDGROUPS * OUTPUTS_PER_SIMDGROUP) + simd_group * OUTPUTS_PER_SIMDGROUP;

  const int expert = expert_ids[route_idx];
  if (expert < 0 || uint(expert) >= expert_count) {
    if (simd_lane == 0) {
      METAL_PRAGMA_UNROLL
      for (uint row = 0; row < OUTPUTS_PER_SIMDGROUP; row++) {
        const uint out_row = out_row_base + row;
        if (out_row < output_width) {
          d[size_t(route_idx) * size_t(output_width) + size_t(out_row)] = DT(0);
        }
      }
    }
    return;
  }
  const uint matrix = uint(expert);
  const uint a_row = input_is_route_major ? route_idx : route_idx / routes_per_token;

  const float outer_scale = float(outer_scales[matrix]);
  const uint block_count = in_vec_size / 32;
  const size_t code_row_bytes = size_t(in_vec_size / 2);
  const size_t scale_row_bytes = size_t(in_vec_size / GROUP_SIZE);

  const device uint8_t* code_bank = codes + size_t(matrix) * size_t(out_vec_size) * code_row_bytes;
  const device uint8_t* scale_bank = scales + size_t(matrix) * size_t(out_vec_size) * scale_row_bytes;
  const device AT* activation = a + size_t(a_row) * size_t(in_vec_size);

  const uint block_lane = simd_lane >> 1;
  const uint block_half = simd_lane & 1u;

  float row_sum[ROWS_PER_SIMDGROUP] = {0.0f};

  for (uint block = block_lane; block < block_count; block += 16) {
    const uint k_base = block * 32 + block_half * 16;
    const device AT* activation_ptr = activation + k_base;

    float4 y0;
    float4 y1;
    float4 y2;
    float4 y3;
    if constexpr (sizeof(AT) == 2) {
      const device bfloat4* vectors = reinterpret_cast<const device bfloat4*>(activation_ptr);
      y0 = static_cast<float4>(vectors[0]);
      y1 = static_cast<float4>(vectors[1]);
      y2 = static_cast<float4>(vectors[2]);
      y3 = static_cast<float4>(vectors[3]);
    } else {
      const device float4* vectors = reinterpret_cast<const device float4*>(activation_ptr);
      y0 = vectors[0];
      y1 = vectors[1];
      y2 = vectors[2];
      y3 = vectors[3];
    }

    METAL_PRAGMA_UNROLL
    for (uint row = 0; row < ROWS_PER_SIMDGROUP; row++) {
      uint weight_row;
      if constexpr (FUSED_GATE_UP) {
        // First half of the slots accumulate value rows, second half gates.
        const uint hidden = out_row_base + row % OUTPUTS_PER_SIMDGROUP;
        if (hidden >= hidden_dim) {
          break;
        }
        weight_row = row < OUTPUTS_PER_SIMDGROUP ? hidden : hidden_dim + hidden;
      } else {
        weight_row = out_row_base + row;
        if (weight_row >= out_vec_size) {
          break;
        }
      }
      const device uint8_t* row_codes = code_bank + size_t(weight_row) * code_row_bytes;
      const uint2 packed = *reinterpret_cast<const device uint2*>(row_codes + size_t(block * 16 + block_half * 8));
      const float4 d0 = float4(e2m1_lut[packed.x & 0xFu], e2m1_lut[(packed.x >> 4u) & 0xFu], e2m1_lut[(packed.x >> 8u) & 0xFu], e2m1_lut[(packed.x >> 12u) & 0xFu]);
      const float4 d1 = float4(e2m1_lut[(packed.x >> 16u) & 0xFu], e2m1_lut[(packed.x >> 20u) & 0xFu], e2m1_lut[(packed.x >> 24u) & 0xFu], e2m1_lut[packed.x >> 28u]);
      const float4 d2 = float4(e2m1_lut[packed.y & 0xFu], e2m1_lut[(packed.y >> 4u) & 0xFu], e2m1_lut[(packed.y >> 8u) & 0xFu], e2m1_lut[(packed.y >> 12u) & 0xFu]);
      const float4 d3 = float4(e2m1_lut[(packed.y >> 16u) & 0xFu], e2m1_lut[(packed.y >> 20u) & 0xFu], e2m1_lut[(packed.y >> 24u) & 0xFu], e2m1_lut[packed.y >> 28u]);

      const device uint8_t* row_scales = scale_bank + size_t(weight_row) * scale_row_bytes;
      const uint scale_index = GROUP_SIZE == 16 ? block * 2 + block_half : block;
      const float block_scale = uzu::gemm::decode_e8m0(row_scales[scale_index]) * outer_scale;

      const float4 partial = y0 * d0 + y1 * d1 + y2 * d2 + y3 * d3;
      row_sum[row] += block_scale * ((partial.x + partial.y) + (partial.z + partial.w));
    }
  }

  const device BT* matrix_biases = expert_bias ? expert_biases + size_t(matrix) * size_t(out_vec_size) : nullptr;
  device DT* output_row = d + size_t(route_idx) * size_t(output_width);
  if constexpr (FUSED_GATE_UP) {
    const float alpha = custom_activation_alpha ? activation_alpha : 1.0f;
    METAL_PRAGMA_UNROLL
    for (uint slot = 0; slot < OUTPUTS_PER_SIMDGROUP; slot++) {
      const float value_sum = simd_sum(row_sum[slot]);
      const float gate_sum = simd_sum(row_sum[slot + OUTPUTS_PER_SIMDGROUP]);
      const uint hidden = out_row_base + slot;
      if (simd_lane == 0 && hidden < hidden_dim) {
        float value = value_sum;
        float gate = gate_sum;
        if (expert_bias) {
          value += float(matrix_biases[hidden]);
          gate += float(matrix_biases[hidden_dim + hidden]);
        }
        if (clip_value) {
          value = clamp(value, value_clip_min, value_clip_max);
        }
        if (clip_gate) {
          gate = clamp(gate, gate_clip_min, gate_clip_max);
        }
        output_row[hidden] = DT(value * activate_silu_alpha(gate, alpha));
      }
    }
    return;
  }
  METAL_PRAGMA_UNROLL
  for (uint row = 0; row < ROWS_PER_SIMDGROUP; row++) {
    const float total = simd_sum(row_sum[row]);
    const uint out_row = out_row_base + row;
    if (simd_lane == 0 && out_row < out_vec_size) {
      float value = total;
      if (expert_bias) {
        value += float(matrix_biases[out_row]);
      }
      output_row[out_row] = DT(value);
    }
  }
}
