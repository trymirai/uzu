#include <metal_stdlib>

#include "../common/defines.h"
#include "../common/dsl.h"
#include "../generated/moe.h"
#include "expert_common.h"

using namespace metal;
using namespace uzu::moe;

#define BATCH_SIMDGROUPS 8
#define BATCH_UP_HIDDEN 4
#define BATCH_UP_LANES_PER_ROW (METAL_SIMD_SIZE / BATCH_UP_HIDDEN)

static_assert(BATCH_SIMDGROUPS == MOE_GEMV_SIMDGROUPS, "the host sizes the grid from the shared constants");
static_assert(BATCH_UP_HIDDEN == MOE_UP_TILE_HIDDEN, "the host sizes the grid from the shared constants");

namespace {

// One hidden unit's affine 4-bit up and gate rows, with their (scale, bias) pairs: bfloat16 scale low, bias high.
struct BatchUpRows {
  const device uint4* up_codes;
  const device uint4* gate_codes;
  const device uint* up_params;
  const device uint* gate_params;
};

// The up and gate dot products of one hidden unit with NB inputs over this lane's chunks: each code is read once
// for all NB inputs.
template <typename T, uint NB>
METAL_FUNC void batch_up_dot(
    const thread BatchUpRows& rows,
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
  for (uint chunk = first_chunk; chunk < chunks; chunk += BATCH_UP_LANES_PER_ROW) {
    const uint4 up_codes = rows.up_codes[chunk];
    const uint4 gate_codes = rows.gate_codes[chunk];
    // A chunk is half of a 64-value group.
    const uint up_pair = rows.up_params[chunk / 2];
    const uint gate_pair = rows.gate_params[chunk / 2];
    const float up_scale = as_type<float>(up_pair << 16);
    const float up_bias = as_type<float>(up_pair & 0xFFFF0000u);
    const float gate_scale = as_type<float>(gate_pair << 16);
    const float gate_bias = as_type<float>(gate_pair & 0xFFFF0000u);
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
  METAL_PRAGMA_UNROLL
  for (uint n = 0; n < NB; ++n) {
    METAL_PRAGMA_UNROLL
    for (ushort offset = 1; offset < BATCH_UP_LANES_PER_ROW; offset <<= 1) {
      up[n] += simd_shuffle_xor(up[n], offset);
      gate[n] += simd_shuffle_xor(gate[n], offset);
    }
  }
}

struct BatchUpActivation {
  bool cap_up;
  float up_soft_cap;
  bool cap_gate;
  float gate_soft_cap;
  float alpha;
};

// Routes [index, index + NB) of the expert: hidden[route] = route_weight * act(up(x), gate(x)) for this lane's
// hidden unit.
template <typename T, uint NB, typename RouteAt>
METAL_FUNC void batch_up_pass(
    const thread BatchUpRows& rows,
    RouteAt route_at,
    const uint index,
    const device T* input,
    const device float* route_weights,
    device half* hidden,
    const uint model_dim,
    const uint hidden_dim,
    const uint hidden_index,
    const uint top_k,
    const uint chunks,
    const uint first_chunk,
    const BatchUpActivation activation
) {
  uint routes[NB];
  const device T* inputs[NB];
  METAL_PRAGMA_UNROLL
  for (uint n = 0; n < NB; ++n) {
    routes[n] = route_at(index + n);
    inputs[n] = input + size_t(routes[n] / top_k) * model_dim;
  }
  float up[NB];
  float gate[NB];
  batch_up_dot<T, NB>(rows, inputs, chunks, first_chunk, up, gate);
  if (first_chunk == 0) {
    METAL_PRAGMA_UNROLL
    for (uint n = 0; n < NB; ++n) {
      const float value = expert_activation(
          up[n],
          gate[n],
          activation.cap_up,
          activation.up_soft_cap,
          activation.cap_gate,
          activation.gate_soft_cap,
          activation.alpha
      );
      hidden[size_t(routes[n]) * hidden_dim + hidden_index] = half(value * route_weights[routes[n]]);
    }
  }
}

} // namespace

// Gate and up projections of affine 4-bit experts for a batch of a few tokens, with the activation and the route
// weight applied: hidden[route] = route_weight * act(up_e(x), gate_e(x)) in half. Simdgroup `simdgroup` of
// threadgroup (tile, slot_group) takes active expert slot_group * BATCH_SIMDGROUPS + simdgroup for the tile's
// BATCH_UP_HIDDEN hidden units and streams its rows once for up to MAX_PASS of the expert's tokens at a time.
// Requires model_dim % (32 * BATCH_UP_LANES_PER_ROW) == 0, hidden_dim % BATCH_UP_HIDDEN == 0 and the MLX affine
// layout of 4-bit codes with one packed (scale, bias) pair per 64 inputs.
template <typename T, uint MAX_PASS>
VARIANTS(T, bfloat)
VARIANTS(MAX_PASS, 2, 4)
PUBLIC KERNEL(MoeExpertUpBatch)(
    const device T* input,
    const device uint8_t* weights,
    const device uint* params,
    const device uint* expert_routes,
    const device uint* active_experts,
    const device float* route_weights,
    device half* hidden,
    const constant uint& tokens,
    const constant uint& model_dim,
    const constant uint& hidden_dim,
    const constant uint& experts,
    const constant uint& top_k,
    const constant float& up_soft_cap,
    const constant float& gate_soft_cap,
    const constant float& activation_alpha,
    const bool cap_up SPECIALIZE,
    const bool cap_gate SPECIALIZE,
    const uint tile GROUPS(hidden_dim / BATCH_UP_HIDDEN),
    const uint slot_group GROUPS((tokens * top_k).min(experts).div_ceil(BATCH_SIMDGROUPS)),
    const uint lane THREADS(METAL_SIMD_SIZE),
    const uint simdgroup THREADS(BATCH_SIMDGROUPS)
) {
  const uint slot = slot_group * BATCH_SIMDGROUPS + simdgroup;
  // The entry load does not wait for the count: the table holds an entry slot per expert.
  const uint active = active_experts[0];
  const uint packed_entry = active_experts[1 + min(slot, experts - 1)];
  if (slot >= active) {
    return;
  }
  const ActiveExpert entry = decode_active_entry(packed_entry);
  const device uint* routes = expert_routes + entry.expert * tokens;
  const auto route_at = [&](uint index) { return index == 0 ? entry.first_route : routes[index]; };

  const uint unit = lane / BATCH_UP_LANES_PER_ROW;
  const uint first_chunk = lane % BATCH_UP_LANES_PER_ROW;
  const uint hidden_index = tile * BATCH_UP_HIDDEN + unit;
  const uint chunks = model_dim / CHUNK_VALUES;
  const uint row_code_bytes = model_dim / 2;
  const uint row_groups = model_dim / AFFINE_GROUP_SIZE;
  const size_t up_row = size_t(entry.expert) * 2 * hidden_dim + hidden_index;
  const size_t gate_row = up_row + hidden_dim;
  const BatchUpRows rows = {
      reinterpret_cast<const device uint4*>(weights + up_row * row_code_bytes),
      reinterpret_cast<const device uint4*>(weights + gate_row * row_code_bytes),
      params + up_row * row_groups,
      params + gate_row * row_groups,
  };
  const BatchUpActivation activation = {cap_up, up_soft_cap, cap_gate, gate_soft_cap, activation_alpha};

  // Passes of MAX_PASS tokens, then one pass for the rest: each pass streams the rows once.
  uint index = 0;
  for (; index + MAX_PASS <= entry.count; index += MAX_PASS) {
    batch_up_pass<T, MAX_PASS>(
        rows,
        route_at,
        index,
        input,
        route_weights,
        hidden,
        model_dim,
        hidden_dim,
        hidden_index,
        top_k,
        chunks,
        first_chunk,
        activation
    );
  }
  const uint remaining = entry.count - index;
  if (MAX_PASS > 3 && remaining == 3) {
    batch_up_pass<T, 3>(
        rows,
        route_at,
        index,
        input,
        route_weights,
        hidden,
        model_dim,
        hidden_dim,
        hidden_index,
        top_k,
        chunks,
        first_chunk,
        activation
    );
  } else if (MAX_PASS > 2 && remaining == 2) {
    batch_up_pass<T, 2>(
        rows,
        route_at,
        index,
        input,
        route_weights,
        hidden,
        model_dim,
        hidden_dim,
        hidden_index,
        top_k,
        chunks,
        first_chunk,
        activation
    );
  } else if (remaining == 1) {
    batch_up_pass<T, 1>(
        rows,
        route_at,
        index,
        input,
        route_weights,
        hidden,
        model_dim,
        hidden_dim,
        hidden_index,
        top_k,
        chunks,
        first_chunk,
        activation
    );
  }
}
