#include <metal_stdlib>

#include "../common/defines.h"
#include "../common/dsl.h"
#include "../generated/moe.h"
#include "expert_common.h"
#include "grouped_affine.h"

using namespace metal;
using namespace uzu::moe;

// Hidden units of one tile: simdgroup j owns 16 units, whose up and gate rows it multiplies together.
#define GROUPED_AFFINE_UP_HIDDEN (GROUPED_AFFINE_BN / 2)

// Gate and up projections of affine 4-bit experts for a large batch on the matrix units, with the activation and
// the route weight applied: hidden[route] = route_weight * act(up_e(x), gate_e(x)) in half. Threadgroup
// (expert, tile) dequantizes the up and gate rows of GROUPED_AFFINE_UP_HIDDEN hidden units to bfloat once per step
// and runs them against the expert's routes, GROUPED_AFFINE_PASS_ROUTES per pass; the bfloat inputs are staged as
// they are. Requires model_dim % GROUPED_AFFINE_BK == 0 and hidden_dim % GROUPED_AFFINE_UP_HIDDEN == 0.
template <typename T>
VARIANTS(T, bfloat)
PUBLIC KERNEL(MoeGroupedAffineUp)(
    const device T* input,
    const device uint8_t* weights,
    const device uint* params,
    const device uint* expert_counts,
    const device uint* expert_routes,
    const device float* route_weights,
    device half* hidden,
    const constant uint& tokens,
    const constant uint& experts,
    const constant uint& model_dim,
    const constant uint& hidden_dim,
    const constant uint& top_k,
    const constant float& up_soft_cap,
    const constant float& gate_soft_cap,
    const constant float& activation_alpha,
    const bool cap_up SPECIALIZE,
    const bool cap_gate SPECIALIZE,
    threadgroup bfloat input_tile[GROUPED_AFFINE_PASS_ROUTES * GROUPED_AFFINE_STRIDE],
    threadgroup bfloat weight_tile[GROUPED_AFFINE_BN * GROUPED_AFFINE_STRIDE],
    threadgroup uint tile_routes[GROUPED_AFFINE_PASS_ROUTES],
    const uint expert GROUPS(experts),
    const uint tile GROUPS(hidden_dim / GROUPED_AFFINE_UP_HIDDEN),
    const uint lane THREADS(METAL_SIMD_SIZE),
    const uint simdgroup THREADS(GROUPED_AFFINE_SIMDGROUPS)
) {
  const uint count = expert_counts[expert];
  if (count == 0) {
    return;
  }
  const uint thread_index = simdgroup * METAL_SIMD_SIZE + lane;
  const uint simdgroup_n = simdgroup % GROUPED_AFFINE_SIMDGROUPS_N;
  const uint simdgroup_m = simdgroup / GROUPED_AFFINE_SIMDGROUPS_N;
  const uint first_hidden = tile * GROUPED_AFFINE_UP_HIDDEN;

  // This thread stages half of weight row `stage_row` of the tile: rows [32 n, 32 n + 16) are the up rows of hidden
  // units [16 n, 16 n + 16) and the next 16 rows their gate rows.
  const uint stage_row = thread_index / 2;
  const uint stage_half = thread_index % 2;
  const uint unit_in_tile = (stage_row / 32) * 16 + stage_row % 16;
  const bool stage_gate = (stage_row % 32) >= 16;
  const size_t weight_row =
      size_t(expert) * 2 * hidden_dim + (stage_gate ? hidden_dim : 0u) + first_hidden + unit_in_tile;
  const device uint2* row_codes = reinterpret_cast<const device uint2*>(weights + weight_row * (model_dim / 2));
  const device uint* row_params = params + weight_row * (model_dim / AFFINE_GROUP_SIZE);

  const uint input_row = thread_index / 4;
  const uint input_part = thread_index % 4;
  const device uint* routes = expert_routes + expert * tokens;
  threadgroup bfloat* input_destination = input_tile + input_row * GROUPED_AFFINE_STRIDE + 8 * input_part;
  threadgroup bfloat* weight_destination = weight_tile + stage_row * GROUPED_AFFINE_STRIDE + 16 * stage_half;

  for (uint first_route = 0; first_route < count; first_route += GROUPED_AFFINE_PASS_ROUTES) {
    const uint rows = min(uint(GROUPED_AFFINE_PASS_ROUTES), count - first_route);
    const uint mtiles = (rows + GROUPED_AFFINE_MT - 1) / GROUPED_AFFINE_MT;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < GROUPED_AFFINE_PASS_ROUTES) {
      tile_routes[thread_index] = thread_index < rows ? routes[first_route + thread_index] : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool input_valid = input_row < rows;
    const device T* input_source = input + size_t(tile_routes[input_row] / top_k) * model_dim + 8 * input_part;

    GroupedAffineAccumulator accumulators[GROUPED_AFFINE_MTILES_PER_SIMDGROUP];
    grouped_affine_pass(
        [&](uint k) {
          GroupedAffineFetch fetch;
          fetch.input = input_valid ? *reinterpret_cast<const device uint4*>(input_source + k) : uint4(0);
          fetch.codes = row_codes[(k + 16 * stage_half) / 16];
          fetch.params = row_params[k / AFFINE_GROUP_SIZE];
          return fetch;
        },
        [&](const thread GroupedAffineFetch& fetch) {
          *reinterpret_cast<threadgroup uint4*>(input_destination) = fetch.input;
          store_affine_weights16(fetch, weight_destination);
        },
        model_dim,
        mtiles,
        input_tile,
        weight_tile,
        simdgroup_n,
        simdgroup_m,
        ushort(lane),
        accumulators
    );

    METAL_PRAGMA_UNROLL
    for (uint j = 0; j < GROUPED_AFFINE_MTILES_PER_SIMDGROUP; ++j) {
      const uint i = simdgroup_m + j * GROUPED_AFFINE_SIMDGROUPS_M;
      if (i < mtiles) {
        for_each_affine_pair(accumulators[j], ushort(lane), [&](ushort row, ushort column, float up, float gate) {
          const uint pass_row = i * GROUPED_AFFINE_MT + row;
          if (pass_row < rows) {
            const uint route = tile_routes[pass_row];
            const float value =
                expert_activation(up, gate, cap_up, up_soft_cap, cap_gate, gate_soft_cap, activation_alpha);
            const uint unit = first_hidden + simdgroup_n * 16 + column;
            hidden[size_t(route) * hidden_dim + unit] = half(value * route_weights[route]);
          }
        });
      }
    }
  }
}
