#include <metal_stdlib>

#include "../common/defines.h"
#include "../common/dsl.h"
#include "../generated/moe.h"
#include "expert_common.h"

using namespace metal;
using namespace uzu::moe;

#define UP_SIMDGROUPS 8
#define UP_TILE_HIDDEN 4
#define UP_LANES_PER_ROW (METAL_SIMD_SIZE / UP_TILE_HIDDEN)

static_assert(UP_SIMDGROUPS == MOE_GEMV_SIMDGROUPS, "the host sizes the grid from the shared constants");
static_assert(UP_TILE_HIDDEN == MOE_UP_TILE_HIDDEN, "the host sizes the grid from the shared constants");

struct UpRows {
  const device uint4* up_codes;
  const device uint4* gate_codes;
  const device uchar* up_scales;
  const device uchar* gate_scales;
  float output_scale;
};

// Accumulates the up and gate rows of one hidden unit against NB routed tokens over this lane's chunks.
template <typename T, uint NB>
METAL_FUNC void up_rows_dot(
    const MoeExpertFormat format,
    const UpRows rows,
    const device T* const (&inputs)[NB],
    const uint chunks,
    const uint first_chunk,
    thread float (&up)[NB],
    thread float (&gate)[NB]
) {
  METAL_PRAGMA_UNROLL
  for (uint n = 0; n < NB; ++n) {
    up[n] = 0.0f;
    gate[n] = 0.0f;
  }
#pragma clang loop unroll_count(3)
  for (uint chunk = first_chunk; chunk < chunks; chunk += UP_LANES_PER_ROW) {
    const uint4 up_codes = rows.up_codes[chunk];
    const uint4 gate_codes = rows.gate_codes[chunk];
    if (format == MoeExpertFormat::Nvfp4) {
      const float2 up_scale = decode_e4m3x2(uint(reinterpret_cast<const device ushort*>(rows.up_scales)[chunk]));
      const float2 gate_scale = decode_e4m3x2(uint(reinterpret_cast<const device ushort*>(rows.gate_scales)[chunk]));
      float up_part[NB][2];
      float gate_part[NB][2];
      METAL_PRAGMA_UNROLL
      for (uint n = 0; n < NB; ++n) {
        up_part[n][0] = up_part[n][1] = gate_part[n][0] = gate_part[n][1] = 0.0f;
      }
      METAL_PRAGMA_UNROLL
      for (uint word = 0; word < 4; ++word) {
        half2 up_pairs[4];
        half2 gate_pairs[4];
        decode_e2m1x8(up_codes[word], up_pairs);
        decode_e2m1x8(gate_codes[word], gate_pairs);
        METAL_PRAGMA_UNROLL
        for (uint n = 0; n < NB; ++n) {
          float values[8];
          load_values8(inputs[n] + chunk * CHUNK_VALUES + 8 * word, values);
          float up_acc = up_part[n][word / 2];
          float gate_acc = gate_part[n][word / 2];
          METAL_PRAGMA_UNROLL
          for (uint k = 0; k < 4; ++k) {
            up_acc = fma(float(up_pairs[k].x), values[k], up_acc);
            up_acc = fma(float(up_pairs[k].y), values[k + 4], up_acc);
            gate_acc = fma(float(gate_pairs[k].x), values[k], gate_acc);
            gate_acc = fma(float(gate_pairs[k].y), values[k + 4], gate_acc);
          }
          up_part[n][word / 2] = up_acc;
          gate_part[n][word / 2] = gate_acc;
        }
      }
      METAL_PRAGMA_UNROLL
      for (uint n = 0; n < NB; ++n) {
        up[n] = fma(up_part[n][0], up_scale.x, fma(up_part[n][1], up_scale.y, up[n]));
        gate[n] = fma(gate_part[n][0], gate_scale.x, fma(gate_part[n][1], gate_scale.y, gate[n]));
      }
    } else {
      // A chunk is half of a 64-value group; the pair packs the bfloat16 scale low and the bias high.
      const uint up_params = reinterpret_cast<const device uint*>(rows.up_scales)[chunk / 2];
      const uint gate_params = reinterpret_cast<const device uint*>(rows.gate_scales)[chunk / 2];
      const float up_scale = as_type<float>(up_params << 16);
      const float up_bias = as_type<float>(up_params & 0xFFFF0000u);
      const float gate_scale = as_type<float>(gate_params << 16);
      const float gate_bias = as_type<float>(gate_params & 0xFFFF0000u);
      METAL_PRAGMA_UNROLL
      for (uint n = 0; n < NB; ++n) {
        float up_low = 0.0f, up_high = 0.0f, gate_low = 0.0f, gate_high = 0.0f, sum = 0.0f;
        METAL_PRAGMA_UNROLL
        for (uint word = 0; word < 4; ++word) {
          float values[8];
          load_values8(inputs[n] + chunk * CHUNK_VALUES + 8 * word, values);
          dot_u4x8(up_codes[word], values, up_low, up_high);
          dot_u4x8(gate_codes[word], values, gate_low, gate_high);
          sum +=
              ((values[0] + values[1]) + (values[2] + values[3])) + ((values[4] + values[5]) + (values[6] + values[7]));
        }
        up[n] = fma(up_scale, fma(up_high, 0.0625f, up_low), fma(up_bias, sum, up[n]));
        gate[n] = fma(gate_scale, fma(gate_high, 0.0625f, gate_low), fma(gate_bias, sum, gate[n]));
      }
    }
  }
  METAL_PRAGMA_UNROLL
  for (uint n = 0; n < NB; ++n) {
    METAL_PRAGMA_UNROLL
    for (ushort offset = 1; offset < UP_LANES_PER_ROW; offset <<= 1) {
      up[n] += simd_shuffle_xor(up[n], offset);
      gate[n] += simd_shuffle_xor(gate[n], offset);
    }
    up[n] *= rows.output_scale;
    gate[n] *= rows.output_scale;
  }
}

// Gate and up projections of the routed experts for a small batch, with the activation and the route weight
// applied: hidden[route] = route_weight * act(up_e(x), gate_e(x)) in half. The up matrix of an expert stacks its
// hidden_dim up rows over its hidden_dim gate rows. Every simdgroup produces the threadgroup's UP_TILE_HIDDEN hidden
// units for its share of the active experts, reading each expert's rows once for all of the expert's tokens.
// With MAX_TOKENS == 1 (decode) the two-token path is compiled out, which keeps register use and so occupancy up.
// Requires tokens <= MAX_TOKENS, model_dim % (32 * UP_LANES_PER_ROW) == 0 and hidden_dim % UP_TILE_HIDDEN == 0.
template <typename T, uint MAX_TOKENS>
VARIANTS(T, bfloat)
VARIANTS(MAX_TOKENS, 1, 32)
PUBLIC KERNEL(MoeExpertUp)(
    const device T* input,
    const device uint8_t* weights,
    const device uint8_t* scales,
    const device float* global_scales OPTIONAL(format == MoeExpertFormat::Nvfp4),
    const device uint* expert_routes,
    const device uint* active_experts,
    const device float* route_weights,
    device half* hidden,
    const constant uint& tokens,
    const constant uint& model_dim,
    const constant uint& hidden_dim,
    const constant uint& top_k,
    const constant float& up_soft_cap,
    const constant float& gate_soft_cap,
    const constant float& activation_alpha,
    const MoeExpertFormat format SPECIALIZE,
    const bool cap_up SPECIALIZE,
    const bool cap_gate SPECIALIZE,
    const uint tile GROUPS(hidden_dim / UP_TILE_HIDDEN),
    const uint lane THREADS(METAL_SIMD_SIZE),
    const uint simdgroup THREADS(UP_SIMDGROUPS)
) {
  const uint unit = lane / UP_LANES_PER_ROW;
  const uint first_chunk = lane % UP_LANES_PER_ROW;
  const uint hidden_index = tile * UP_TILE_HIDDEN + unit;
  const uint chunks = model_dim / CHUNK_VALUES;
  const uint row_code_bytes = model_dim / 2;
  const uint row_scale_bytes =
      format == MoeExpertFormat::Nvfp4 ? model_dim / NVFP4_GROUP_SIZE : model_dim / AFFINE_GROUP_SIZE * 4;
  const uint rows_per_expert = 2 * hidden_dim;
  const uint active = active_experts[0];

  for (uint slot = simdgroup; slot < active; slot += UP_SIMDGROUPS) {
    const ActiveExpert entry = decode_active_entry(active_experts[1 + slot]);
    const uint expert = entry.expert;
    const uint count = entry.count;
    // The first route rides in the entry; later ones come from the expert's route list.
    const device uint* routes = expert_routes + expert * tokens;
    const auto route_at = [&](uint index) { return index == 0 ? entry.first_route : routes[index]; };
    const size_t up_row = size_t(expert) * rows_per_expert + hidden_index;
    const size_t gate_row = up_row + hidden_dim;
    const UpRows rows = {
        reinterpret_cast<const device uint4*>(weights + up_row * row_code_bytes),
        reinterpret_cast<const device uint4*>(weights + gate_row * row_code_bytes),
        scales + up_row * row_scale_bytes,
        scales + gate_row * row_scale_bytes,
        format == MoeExpertFormat::Nvfp4 ? global_scales[expert] * NVFP4_DECODE_SCALE : 1.0f,
    };

    uint index = 0;
    for (; MAX_TOKENS > 1 && index + 2 <= count; index += 2) {
      const uint route_pair[2] = {route_at(index), route_at(index + 1)};
      const device T* const inputs[2] = {
          input + size_t(route_pair[0] / top_k) * model_dim,
          input + size_t(route_pair[1] / top_k) * model_dim,
      };
      float up[2];
      float gate[2];
      up_rows_dot<T, 2>(format, rows, inputs, chunks, first_chunk, up, gate);
      if (first_chunk == 0) {
        METAL_PRAGMA_UNROLL
        for (uint n = 0; n < 2; ++n) {
          const float value =
              expert_activation(up[n], gate[n], cap_up, up_soft_cap, cap_gate, gate_soft_cap, activation_alpha);
          hidden[size_t(route_pair[n]) * hidden_dim + hidden_index] = half(value * route_weights[route_pair[n]]);
        }
      }
    }
    if (index < count) {
      const uint route = route_at(index);
      const device T* const inputs[1] = {input + size_t(route / top_k) * model_dim};
      float up[1];
      float gate[1];
      up_rows_dot<T, 1>(format, rows, inputs, chunks, first_chunk, up, gate);
      if (first_chunk == 0) {
        const float value =
            expert_activation(up[0], gate[0], cap_up, up_soft_cap, cap_gate, gate_soft_cap, activation_alpha);
        hidden[size_t(route) * hidden_dim + hidden_index] = half(value * route_weights[route]);
      }
    }
  }
}
