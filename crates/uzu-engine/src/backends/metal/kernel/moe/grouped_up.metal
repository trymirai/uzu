#include <metal_stdlib>

#include "../common/defines.h"
#include "../common/dsl.h"
#include "../generated/moe.h"
#include "expert_common.h"
#include "grouped_common.h"

using namespace metal;
using namespace uzu::moe;

// Hidden units of one tile: every simdgroup column owns 16 units, whose up and gate rows it multiplies together.
#define GROUPED_UP_TILE_HIDDEN (GROUPED_BN / 2)

// Gate and up projections of the routed experts for a large batch on the matrix units, with the activation and the
// route weight applied: hidden[route] = route_weight * act(up_e(x), gate_e(x)) in half. Threadgroup (expert, tile)
// walks the expert's routes GROUPED_BM at a time for GROUPED_UP_TILE_HIDDEN hidden units; the weights are
// dequantized to bfloat in threadgroup memory, exactly for NVFP4, and the bfloat inputs are staged as they are.
// Requires model_dim % GROUPED_BK == 0 and hidden_dim % GROUPED_UP_TILE_HIDDEN == 0.
template <typename T>
VARIANTS(T, bfloat)
PUBLIC KERNEL(MoeGroupedUp)(
    const device T* input,
    const device uint8_t* weights,
    const device uint8_t* scales,
    const device float* global_scales OPTIONAL(format == MoeExpertFormat::Nvfp4),
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
    const MoeExpertFormat format SPECIALIZE,
    const bool cap_up SPECIALIZE,
    const bool cap_gate SPECIALIZE,
    threadgroup bfloat input_tile[GROUPED_BM * GROUPED_STRIDE],
    threadgroup bfloat weight_tile[GROUPED_BN * GROUPED_STRIDE],
    threadgroup uint tile_routes[GROUPED_BM],
    const uint expert GROUPS(experts),
    const uint tile GROUPS(hidden_dim / GROUPED_UP_TILE_HIDDEN),
    const uint lane THREADS(METAL_SIMD_SIZE),
    const uint simdgroup THREADS(GROUPED_SIMDGROUPS)
) {
  const uint count = expert_counts[expert];
  if (count == 0) {
    return;
  }
  const uint thread_index = simdgroup * METAL_SIMD_SIZE + lane;
  const uint simdgroup_m = simdgroup / 4;
  const uint simdgroup_n = simdgroup % 4;
  const uint first_hidden = tile * GROUPED_UP_TILE_HIDDEN;

  // This thread stages weight row `stage_row` of the tile: rows [32 j, 32 j + 16) are the up rows of hidden units
  // [16 j, 16 j + 16) and the next 16 rows their gate rows.
  const uint stage_row = thread_index / 2;
  const uint stage_half = thread_index % 2;
  const uint unit_in_tile = (stage_row / 32) * 16 + stage_row % 16;
  const bool stage_gate = (stage_row % 32) >= 16;
  const size_t weight_row =
      size_t(expert) * 2 * hidden_dim + (stage_gate ? hidden_dim : 0u) + first_hidden + unit_in_tile;
  const uint row_scale_bytes =
      format == MoeExpertFormat::Nvfp4 ? model_dim / NVFP4_GROUP_SIZE : model_dim / AFFINE_GROUP_SIZE * 4;
  const device uchar* row_codes = weights + weight_row * (model_dim / 2);
  const device uchar* row_scales = scales + weight_row * row_scale_bytes;
  const float output_scale = format == MoeExpertFormat::Nvfp4 ? global_scales[expert] : 1.0f;

  const uint input_row = thread_index / 4;
  const uint input_part = thread_index % 4;
  const device uint* routes = expert_routes + expert * tokens;

  for (uint first_route = 0; first_route < count; first_route += GROUPED_BM) {
    const uint rows = min(uint(GROUPED_BM), count - first_route);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < GROUPED_BM) {
      tile_routes[thread_index] = thread_index < rows ? routes[first_route + thread_index] : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool input_valid = input_row < rows;
    const device T* input_source = input + size_t(tile_routes[input_row] / top_k) * model_dim + 8 * input_part;

    threadgroup bfloat* input_destination = input_tile + input_row * GROUPED_STRIDE + 8 * input_part;
    threadgroup bfloat* weight_destination = weight_tile + stage_row * GROUPED_STRIDE + 16 * stage_half;
    GroupedAccumulator accumulator;
    grouped_tile_mma(
        [&](uint k) {
          GroupedFetch fetch;
          fetch.input = input_valid ? *reinterpret_cast<const device uint4*>(input_source + k) : uint4(0);
          fetch_weights16(format, row_codes, row_scales, k + 16 * stage_half, fetch);
          return fetch;
        },
        [&](const thread GroupedFetch& fetch) {
          *reinterpret_cast<threadgroup uint4*>(input_destination) = fetch.input;
          store_weights16(format, fetch, weight_destination);
        },
        model_dim,
        input_tile,
        weight_tile,
        simdgroup_m,
        simdgroup_n,
        ushort(lane),
        accumulator
    );

    for_each_column_pair(accumulator, ushort(lane), [&](ushort row, ushort column, float up, float gate) {
      const uint tile_row = simdgroup_m * 32 + row;
      if (tile_row < rows) {
        const uint route = tile_routes[tile_row];
        const float value = expert_activation(
            up * output_scale,
            gate * output_scale,
            cap_up,
            up_soft_cap,
            cap_gate,
            gate_soft_cap,
            activation_alpha
        );
        const uint unit = first_hidden + simdgroup_n * 16 + column;
        hidden[size_t(route) * hidden_dim + unit] = half(value * route_weights[route]);
      }
    });
  }
}
