#include <metal_stdlib>

#include "../common/defines.h"
#include "../common/dsl.h"
#include "../generated/moe.h"
#include "expert_common.h"
#include "grouped_common.h"

using namespace metal;
using namespace uzu::moe;

// Down projections of the routed experts for a large batch on the matrix units, one output row per route:
// expert_outputs[route] = down_e(hidden[route]), with the route weight already in `hidden`. Threadgroup
// (expert, tile) walks the expert's routes GROUPED_BM at a time for GROUPED_BN output dimensions.
// Requires hidden_dim % GROUPED_BK == 0 and model_dim % GROUPED_BN == 0.
template <typename T>
VARIANTS(T, bfloat)
PUBLIC KERNEL(MoeGroupedDown)(
    const device half* hidden,
    const device uint8_t* weights,
    const device uint8_t* scales,
    const device float* global_scales OPTIONAL(format == MoeExpertFormat::Nvfp4),
    const device uint* expert_counts,
    const device uint* expert_routes,
    device T* expert_outputs,
    const constant uint& tokens,
    const constant uint& experts,
    const constant uint& model_dim,
    const constant uint& hidden_dim,
    const MoeExpertFormat format SPECIALIZE,
    threadgroup half input_tile[GROUPED_BM * GROUPED_STRIDE],
    threadgroup half weight_tile[GROUPED_BN * GROUPED_STRIDE],
    threadgroup uint tile_routes[GROUPED_BM],
    const uint expert GROUPS(experts),
    const uint tile GROUPS(model_dim / GROUPED_BN),
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
  const uint first_output = tile * GROUPED_BN;

  const uint stage_row = thread_index / 2;
  const uint stage_half = thread_index % 2;
  const size_t weight_row = size_t(expert) * model_dim + first_output + stage_row;
  const uint row_scale_bytes =
      format == MoeExpertFormat::Nvfp4 ? hidden_dim / NVFP4_GROUP_SIZE : hidden_dim / AFFINE_GROUP_SIZE * 4;
  const device uchar* row_codes = weights + weight_row * (hidden_dim / 2);
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
    const device half* input_source = hidden + size_t(tile_routes[input_row]) * hidden_dim + 8 * input_part;

    threadgroup half* input_destination = input_tile + input_row * GROUPED_STRIDE + 8 * input_part;
    threadgroup half* weight_destination = weight_tile + stage_row * GROUPED_STRIDE + 16 * stage_half;
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
        hidden_dim,
        input_tile,
        weight_tile,
        simdgroup_m,
        simdgroup_n,
        ushort(lane),
        accumulator
    );

    for_each_column_pair(accumulator, ushort(lane), [&](ushort row, ushort column, float first, float second) {
      const uint tile_row = simdgroup_m * 32 + row;
      if (tile_row < rows) {
        device T* output_row =
            expert_outputs + size_t(tile_routes[tile_row]) * model_dim + first_output + simdgroup_n * 32 + column;
        output_row[0] = T(first * output_scale);
        output_row[16] = T(second * output_scale);
      }
    });
  }
}
