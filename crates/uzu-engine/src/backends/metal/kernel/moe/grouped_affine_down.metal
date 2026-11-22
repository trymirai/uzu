#include <metal_stdlib>

#include "../common/defines.h"
#include "../common/dsl.h"
#include "../generated/moe.h"
#include "expert_common.h"
#include "grouped_affine.h"

using namespace metal;
using namespace uzu::moe;

// Down projections of affine 4-bit experts for a large batch on the matrix units, one output row per route:
// expert_outputs[route] = down_e(hidden[route]), with the route weight already in `hidden`. Threadgroup
// (expert, tile) dequantizes GROUPED_AFFINE_BN output dimensions to half once per step and runs them against the
// expert's routes, GROUPED_AFFINE_PASS_ROUTES per pass.
// Requires hidden_dim % GROUPED_AFFINE_BK == 0 and model_dim % GROUPED_AFFINE_BN == 0.
template <typename T>
VARIANTS(T, bfloat)
PUBLIC KERNEL(MoeGroupedAffineDown)(
    const device half* hidden,
    const device uint8_t* weights,
    const device uint* params,
    const device uint* expert_counts,
    const device uint* expert_routes,
    device T* expert_outputs,
    const constant uint& tokens,
    const constant uint& experts,
    const constant uint& model_dim,
    const constant uint& hidden_dim,
    threadgroup half input_tile[GROUPED_AFFINE_PASS_ROUTES * GROUPED_AFFINE_STRIDE],
    threadgroup half weight_tile[GROUPED_AFFINE_BN * GROUPED_AFFINE_STRIDE],
    threadgroup uint tile_routes[GROUPED_AFFINE_PASS_ROUTES],
    const uint expert GROUPS(experts),
    const uint tile GROUPS(model_dim / GROUPED_AFFINE_BN),
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
  const uint first_output = tile * GROUPED_AFFINE_BN;
  const uint stage_row = thread_index / 2;
  const uint stage_half = thread_index % 2;
  const size_t weight_row = size_t(expert) * model_dim + first_output + stage_row;
  const device uint2* row_codes = reinterpret_cast<const device uint2*>(weights + weight_row * (hidden_dim / 2));
  const device uint* row_params = params + weight_row * (hidden_dim / AFFINE_GROUP_SIZE);

  const uint input_row = thread_index / 4;
  const uint input_part = thread_index % 4;
  const device uint* routes = expert_routes + expert * tokens;
  threadgroup half* input_destination = input_tile + input_row * GROUPED_AFFINE_STRIDE + 8 * input_part;
  threadgroup half* weight_destination = weight_tile + stage_row * GROUPED_AFFINE_STRIDE + 16 * stage_half;

  for (uint first_route = 0; first_route < count; first_route += GROUPED_AFFINE_PASS_ROUTES) {
    const uint rows = min(uint(GROUPED_AFFINE_PASS_ROUTES), count - first_route);
    const uint mtiles = (rows + GROUPED_AFFINE_MT - 1) / GROUPED_AFFINE_MT;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < GROUPED_AFFINE_PASS_ROUTES) {
      tile_routes[thread_index] = thread_index < rows ? routes[first_route + thread_index] : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool input_valid = input_row < rows;
    const device half* input_source = hidden + size_t(tile_routes[input_row]) * hidden_dim + 8 * input_part;

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
        hidden_dim,
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
        for_each_affine_pair(accumulators[j], ushort(lane), [&](ushort row, ushort column, float first, float second) {
          const uint pass_row = i * GROUPED_AFFINE_MT + row;
          if (pass_row < rows) {
            device T* output_row =
                expert_outputs + size_t(tile_routes[pass_row]) * model_dim + first_output + simdgroup_n * 32 + column;
            output_row[0] = T(first);
            output_row[16] = T(second);
          }
        });
      }
    }
  }
}
