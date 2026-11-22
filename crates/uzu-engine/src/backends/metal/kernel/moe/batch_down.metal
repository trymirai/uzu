#include <metal_stdlib>

#include "../common/defines.h"
#include "../common/dsl.h"
#include "../generated/moe.h"
#include "expert_common.h"

using namespace metal;
using namespace uzu::moe;

#define BATCH_SIMDGROUPS 8
#define BATCH_DOWN_ROWS 16
#define BATCH_DOWN_ROW_GROUPS 8
#define BATCH_DOWN_LANES_PER_ROW (METAL_SIMD_SIZE / BATCH_DOWN_ROW_GROUPS)
#define BATCH_DOWN_ROWS_PER_LANE (BATCH_DOWN_ROWS / BATCH_DOWN_ROW_GROUPS)

static_assert(BATCH_SIMDGROUPS == MOE_GEMV_SIMDGROUPS, "the host sizes the grid from the shared constants");
static_assert(BATCH_DOWN_ROWS == MOE_DOWN_TILE_ROWS, "the host sizes the grid from the shared constants");

namespace {

// This lane's affine 4-bit down rows, with their (scale, bias) pairs: bfloat16 scale low, bias high.
struct BatchDownRows {
  const device uint4* codes[BATCH_DOWN_ROWS_PER_LANE];
  const device uint* params[BATCH_DOWN_ROWS_PER_LANE];
};

// This lane's down rows against NB routes' hidden activations over this lane's chunks: each code is read once for
// all NB routes.
template <uint NB>
METAL_FUNC void batch_down_dot(
    const thread BatchDownRows& rows,
    const device half* const (&inputs)[NB],
    const uint chunks,
    const uint first_chunk,
    thread float (&acc)[NB][BATCH_DOWN_ROWS_PER_LANE]
) {
  METAL_PRAGMA_UNROLL
  for (uint n = 0; n < NB; ++n) {
    METAL_PRAGMA_UNROLL
    for (uint r = 0; r < BATCH_DOWN_ROWS_PER_LANE; ++r) {
      acc[n][r] = 0.0f;
    }
  }
#pragma clang loop unroll_count(3)
  for (uint chunk = first_chunk; chunk < chunks; chunk += BATCH_DOWN_LANES_PER_ROW) {
    uint4 codes[BATCH_DOWN_ROWS_PER_LANE];
    float scale[BATCH_DOWN_ROWS_PER_LANE];
    float bias[BATCH_DOWN_ROWS_PER_LANE];
    METAL_PRAGMA_UNROLL
    for (uint r = 0; r < BATCH_DOWN_ROWS_PER_LANE; ++r) {
      codes[r] = rows.codes[r][chunk];
      // A chunk is half of a 64-value group.
      const uint pair = rows.params[r][chunk / 2];
      scale[r] = as_type<float>(pair << 16);
      bias[r] = as_type<float>(pair & 0xFFFF0000u);
    }
    METAL_PRAGMA_UNROLL
    for (uint n = 0; n < NB; ++n) {
      float low[BATCH_DOWN_ROWS_PER_LANE];
      float high[BATCH_DOWN_ROWS_PER_LANE];
      METAL_PRAGMA_UNROLL
      for (uint r = 0; r < BATCH_DOWN_ROWS_PER_LANE; ++r) {
        low[r] = high[r] = 0.0f;
      }
      float sum = 0.0f;
      METAL_PRAGMA_UNROLL
      for (uint word = 0; word < 4; ++word) {
        float values[8];
        load_values8(inputs[n] + chunk * CHUNK_VALUES + 8 * word, values);
        METAL_PRAGMA_UNROLL
        for (uint r = 0; r < BATCH_DOWN_ROWS_PER_LANE; ++r) {
          dot_u4x8(codes[r][word], values, low[r], high[r]);
        }
        sum +=
            ((values[0] + values[1]) + (values[2] + values[3])) + ((values[4] + values[5]) + (values[6] + values[7]));
      }
      METAL_PRAGMA_UNROLL
      for (uint r = 0; r < BATCH_DOWN_ROWS_PER_LANE; ++r) {
        acc[n][r] = fma(scale[r], fma(high[r], 0.0625f, low[r]), fma(bias[r], sum, acc[n][r]));
      }
    }
  }
  METAL_PRAGMA_UNROLL
  for (uint n = 0; n < NB; ++n) {
    METAL_PRAGMA_UNROLL
    for (uint r = 0; r < BATCH_DOWN_ROWS_PER_LANE; ++r) {
      METAL_PRAGMA_UNROLL
      for (ushort offset = 1; offset < BATCH_DOWN_LANES_PER_ROW; offset <<= 1) {
        acc[n][r] += simd_shuffle_xor(acc[n][r], offset);
      }
    }
  }
}

// Routes [index, index + NB) of the expert: route_outputs[route] = down(hidden[route]) for this lane's rows.
template <uint NB, typename RouteAt>
METAL_FUNC void batch_down_pass(
    const thread BatchDownRows& rows,
    RouteAt route_at,
    const uint index,
    const device half* hidden,
    device float* route_outputs,
    const uint model_dim,
    const uint hidden_dim,
    const uint first_row,
    const uint row_group,
    const uint chunks,
    const uint first_chunk
) {
  uint routes[NB];
  const device half* inputs[NB];
  METAL_PRAGMA_UNROLL
  for (uint n = 0; n < NB; ++n) {
    routes[n] = route_at(index + n);
    inputs[n] = hidden + size_t(routes[n]) * hidden_dim;
  }
  float acc[NB][BATCH_DOWN_ROWS_PER_LANE];
  batch_down_dot<NB>(rows, inputs, chunks, first_chunk, acc);
  if (first_chunk == 0) {
    METAL_PRAGMA_UNROLL
    for (uint n = 0; n < NB; ++n) {
      device float* output_row = route_outputs + size_t(routes[n]) * model_dim + first_row + row_group;
      METAL_PRAGMA_UNROLL
      for (uint r = 0; r < BATCH_DOWN_ROWS_PER_LANE; ++r) {
        output_row[r * BATCH_DOWN_ROW_GROUPS] = acc[n][r];
      }
    }
  }
}

} // namespace

// Down projections of affine 4-bit experts for a batch of a few tokens, one float32 output row per route:
// route_outputs[route] = down_e(hidden[route]), the route weight already in `hidden`. Simdgroup `simdgroup` of
// threadgroup (tile, slot_group) takes active expert slot_group * BATCH_SIMDGROUPS + simdgroup for the tile's
// BATCH_DOWN_ROWS output rows and streams its rows once for up to MAX_PASS of the expert's routes at a time.
// MoeSumRoutes then adds every token's routes.
// Requires hidden_dim % (32 * BATCH_DOWN_LANES_PER_ROW) == 0, model_dim % BATCH_DOWN_ROWS == 0 and the MLX affine
// layout of 4-bit codes with one packed (scale, bias) pair per 64 inputs.
template <uint MAX_PASS>
VARIANTS(MAX_PASS, 2, 4)
PUBLIC KERNEL(MoeExpertDownBatch)(
    const device half* hidden,
    const device uint8_t* weights,
    const device uint* params,
    const device uint* expert_routes,
    const device uint* active_experts,
    device float* route_outputs,
    const constant uint& tokens,
    const constant uint& model_dim,
    const constant uint& hidden_dim,
    const constant uint& experts,
    const constant uint& top_k,
    const uint tile GROUPS(model_dim / BATCH_DOWN_ROWS),
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

  const uint row_group = lane / BATCH_DOWN_LANES_PER_ROW;
  const uint first_chunk = lane % BATCH_DOWN_LANES_PER_ROW;
  const uint chunks = hidden_dim / CHUNK_VALUES;
  const uint row_code_bytes = hidden_dim / 2;
  const uint row_groups = hidden_dim / AFFINE_GROUP_SIZE;
  const uint first_row = tile * BATCH_DOWN_ROWS;
  BatchDownRows rows;
  METAL_PRAGMA_UNROLL
  for (uint r = 0; r < BATCH_DOWN_ROWS_PER_LANE; ++r) {
    const size_t row = size_t(entry.expert) * model_dim + first_row + row_group + r * BATCH_DOWN_ROW_GROUPS;
    rows.codes[r] = reinterpret_cast<const device uint4*>(weights + row * row_code_bytes);
    rows.params[r] = params + row * row_groups;
  }

  // Passes of MAX_PASS routes, then one pass for the rest: each pass streams the rows once.
  uint index = 0;
  for (; index + MAX_PASS <= entry.count; index += MAX_PASS) {
    batch_down_pass<MAX_PASS>(
        rows,
        route_at,
        index,
        hidden,
        route_outputs,
        model_dim,
        hidden_dim,
        first_row,
        row_group,
        chunks,
        first_chunk
    );
  }
  const uint remaining = entry.count - index;
  if (MAX_PASS > 3 && remaining == 3) {
    batch_down_pass<3>(
        rows,
        route_at,
        index,
        hidden,
        route_outputs,
        model_dim,
        hidden_dim,
        first_row,
        row_group,
        chunks,
        first_chunk
    );
  } else if (MAX_PASS > 2 && remaining == 2) {
    batch_down_pass<2>(
        rows,
        route_at,
        index,
        hidden,
        route_outputs,
        model_dim,
        hidden_dim,
        first_row,
        row_group,
        chunks,
        first_chunk
    );
  } else if (remaining == 1) {
    batch_down_pass<1>(
        rows,
        route_at,
        index,
        hidden,
        route_outputs,
        model_dim,
        hidden_dim,
        first_row,
        row_group,
        chunks,
        first_chunk
    );
  }
}
