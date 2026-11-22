#include <metal_stdlib>

#include "../common/defines.h"
#include "../common/dsl.h"
#include "../generated/moe.h"
#include "expert_common.h"

using namespace metal;
using namespace uzu::moe;

#define DOWN_SIMDGROUPS 8
#define DOWN_TILE_ROWS 16
#define DOWN_ROW_GROUPS 8
#define DOWN_LANES_PER_ROW (METAL_SIMD_SIZE / DOWN_ROW_GROUPS)
#define DOWN_ROWS_PER_LANE (DOWN_TILE_ROWS / DOWN_ROW_GROUPS)

static_assert(DOWN_SIMDGROUPS == MOE_GEMV_SIMDGROUPS, "the host sizes the grid from the shared constants");
static_assert(DOWN_TILE_ROWS == MOE_DOWN_TILE_ROWS, "the host sizes the grid from the shared constants");

struct DownRows {
  const device uint4* codes[DOWN_ROWS_PER_LANE];
  const device uchar* scales[DOWN_ROWS_PER_LANE];
  float output_scale;
};

// Accumulates this lane's rows against NB routes' hidden activations over this lane's chunks.
template <uint NB>
METAL_FUNC void down_rows_dot(
    const MoeExpertFormat format,
    const thread DownRows& rows,
    const device half* const (&inputs)[NB],
    const uint chunks,
    const uint first_chunk,
    thread float (&acc)[NB][DOWN_ROWS_PER_LANE]
) {
  METAL_PRAGMA_UNROLL
  for (uint n = 0; n < NB; ++n) {
    METAL_PRAGMA_UNROLL
    for (uint r = 0; r < DOWN_ROWS_PER_LANE; ++r) {
      acc[n][r] = 0.0f;
    }
  }
#pragma clang loop unroll_count(3)
  for (uint chunk = first_chunk; chunk < chunks; chunk += DOWN_LANES_PER_ROW) {
    uint4 codes[DOWN_ROWS_PER_LANE];
    METAL_PRAGMA_UNROLL
    for (uint r = 0; r < DOWN_ROWS_PER_LANE; ++r) {
      codes[r] = rows.codes[r][chunk];
    }
    if (format == MoeExpertFormat::Nvfp4) {
      float2 scale[DOWN_ROWS_PER_LANE];
      METAL_PRAGMA_UNROLL
      for (uint r = 0; r < DOWN_ROWS_PER_LANE; ++r) {
        scale[r] = decode_e4m3x2(uint(reinterpret_cast<const device ushort*>(rows.scales[r])[chunk]));
      }
      float part[NB][DOWN_ROWS_PER_LANE][2];
      METAL_PRAGMA_UNROLL
      for (uint n = 0; n < NB; ++n) {
        METAL_PRAGMA_UNROLL
        for (uint r = 0; r < DOWN_ROWS_PER_LANE; ++r) {
          part[n][r][0] = part[n][r][1] = 0.0f;
        }
      }
      METAL_PRAGMA_UNROLL
      for (uint word = 0; word < 4; ++word) {
        half2 pairs[DOWN_ROWS_PER_LANE][4];
        METAL_PRAGMA_UNROLL
        for (uint r = 0; r < DOWN_ROWS_PER_LANE; ++r) {
          decode_e2m1x8(codes[r][word], pairs[r]);
        }
        METAL_PRAGMA_UNROLL
        for (uint n = 0; n < NB; ++n) {
          float values[8];
          load_values8(inputs[n] + chunk * CHUNK_VALUES + 8 * word, values);
          METAL_PRAGMA_UNROLL
          for (uint r = 0; r < DOWN_ROWS_PER_LANE; ++r) {
            float sum = part[n][r][word / 2];
            METAL_PRAGMA_UNROLL
            for (uint k = 0; k < 4; ++k) {
              sum = fma(float(pairs[r][k].x), values[k], sum);
              sum = fma(float(pairs[r][k].y), values[k + 4], sum);
            }
            part[n][r][word / 2] = sum;
          }
        }
      }
      METAL_PRAGMA_UNROLL
      for (uint n = 0; n < NB; ++n) {
        METAL_PRAGMA_UNROLL
        for (uint r = 0; r < DOWN_ROWS_PER_LANE; ++r) {
          acc[n][r] = fma(part[n][r][0], scale[r].x, fma(part[n][r][1], scale[r].y, acc[n][r]));
        }
      }
    } else {
      float scale[DOWN_ROWS_PER_LANE];
      float bias[DOWN_ROWS_PER_LANE];
      METAL_PRAGMA_UNROLL
      for (uint r = 0; r < DOWN_ROWS_PER_LANE; ++r) {
        const uint params = reinterpret_cast<const device uint*>(rows.scales[r])[chunk / 2];
        scale[r] = as_type<float>(params << 16);
        bias[r] = as_type<float>(params & 0xFFFF0000u);
      }
      METAL_PRAGMA_UNROLL
      for (uint n = 0; n < NB; ++n) {
        float low[DOWN_ROWS_PER_LANE];
        float high[DOWN_ROWS_PER_LANE];
        METAL_PRAGMA_UNROLL
        for (uint r = 0; r < DOWN_ROWS_PER_LANE; ++r) {
          low[r] = high[r] = 0.0f;
        }
        float sum = 0.0f;
        METAL_PRAGMA_UNROLL
        for (uint word = 0; word < 4; ++word) {
          float values[8];
          load_values8(inputs[n] + chunk * CHUNK_VALUES + 8 * word, values);
          METAL_PRAGMA_UNROLL
          for (uint r = 0; r < DOWN_ROWS_PER_LANE; ++r) {
            dot_u4x8(codes[r][word], values, low[r], high[r]);
          }
          sum +=
              ((values[0] + values[1]) + (values[2] + values[3])) + ((values[4] + values[5]) + (values[6] + values[7]));
        }
        METAL_PRAGMA_UNROLL
        for (uint r = 0; r < DOWN_ROWS_PER_LANE; ++r) {
          acc[n][r] = fma(scale[r], fma(high[r], 0.0625f, low[r]), fma(bias[r], sum, acc[n][r]));
        }
      }
    }
  }
  METAL_PRAGMA_UNROLL
  for (uint n = 0; n < NB; ++n) {
    METAL_PRAGMA_UNROLL
    for (uint r = 0; r < DOWN_ROWS_PER_LANE; ++r) {
      METAL_PRAGMA_UNROLL
      for (ushort offset = 1; offset < DOWN_LANES_PER_ROW; offset <<= 1) {
        acc[n][r] += simd_shuffle_xor(acc[n][r], offset);
      }
      acc[n][r] *= rows.output_scale;
    }
  }
}

// Down projections of the routed experts for a small batch, summed over each token's routes:
// output[token] = sum over the token's routes of down_e(hidden[route]). The route weights are already in `hidden`.
// Every simdgroup takes its share of the active experts for the threadgroup's DOWN_TILE_ROWS output rows, reading
// each expert's rows once for all of the expert's tokens; the threadgroup then sums the simdgroups' partial rows.
// With MAX_TOKENS == 1 (decode) the two-token path is compiled out, which keeps register use and so occupancy up.
// Requires tokens <= MAX_TOKENS, hidden_dim % (32 * DOWN_LANES_PER_ROW) == 0 and model_dim % DOWN_TILE_ROWS == 0.
template <typename T, uint MAX_TOKENS>
VARIANTS(T, bfloat)
VARIANTS(MAX_TOKENS, 1, 8, 32)
PUBLIC KERNEL(MoeExpertDown)(
    const device half* hidden,
    const device uint8_t* weights,
    const device uint8_t* scales,
    const device float* global_scales OPTIONAL(format == MoeExpertFormat::Nvfp4),
    const device uint* expert_routes,
    const device uint* active_experts,
    device T* output,
    const constant uint& tokens,
    const constant uint& model_dim,
    const constant uint& hidden_dim,
    const constant uint& top_k,
    const MoeExpertFormat format SPECIALIZE,
    threadgroup float partials[DOWN_SIMDGROUPS * MAX_TOKENS * DOWN_TILE_ROWS],
    const uint tile GROUPS(model_dim / DOWN_TILE_ROWS),
    const uint lane THREADS(METAL_SIMD_SIZE),
    const uint simdgroup THREADS(DOWN_SIMDGROUPS)
) {
  const uint thread_index = simdgroup * METAL_SIMD_SIZE + lane;
  const uint tile_values = tokens * DOWN_TILE_ROWS;
  threadgroup float* simdgroup_partials = partials + simdgroup * MAX_TOKENS * DOWN_TILE_ROWS;
  for (uint index = lane; index < tile_values; index += METAL_SIMD_SIZE) {
    simdgroup_partials[index] = 0.0f;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

  const uint row_group = lane / DOWN_LANES_PER_ROW;
  const uint first_chunk = lane % DOWN_LANES_PER_ROW;
  const uint chunks = hidden_dim / CHUNK_VALUES;
  const uint row_code_bytes = hidden_dim / 2;
  const uint row_scale_bytes =
      format == MoeExpertFormat::Nvfp4 ? hidden_dim / NVFP4_GROUP_SIZE : hidden_dim / AFFINE_GROUP_SIZE * 4;
  const uint active = active_experts[0];

  for (uint slot = simdgroup; slot < active; slot += DOWN_SIMDGROUPS) {
    const ActiveExpert entry = decode_active_entry(active_experts[1 + slot]);
    const uint expert = entry.expert;
    const uint count = entry.count;
    // The first route rides in the entry; later ones come from the expert's route list.
    const device uint* routes = expert_routes + expert * tokens;
    const auto route_at = [&](uint index) { return index == 0 ? entry.first_route : routes[index]; };
    DownRows rows;
    METAL_PRAGMA_UNROLL
    for (uint r = 0; r < DOWN_ROWS_PER_LANE; ++r) {
      const size_t row = size_t(expert) * model_dim + tile * DOWN_TILE_ROWS + row_group + r * DOWN_ROW_GROUPS;
      rows.codes[r] = reinterpret_cast<const device uint4*>(weights + row * row_code_bytes);
      rows.scales[r] = scales + row * row_scale_bytes;
    }
    rows.output_scale = format == MoeExpertFormat::Nvfp4 ? global_scales[expert] * NVFP4_DECODE_SCALE : 1.0f;

    uint index = 0;
    for (; MAX_TOKENS > 1 && index + 2 <= count; index += 2) {
      const uint route_pair[2] = {route_at(index), route_at(index + 1)};
      const device half* const inputs[2] = {
          hidden + size_t(route_pair[0]) * hidden_dim,
          hidden + size_t(route_pair[1]) * hidden_dim,
      };
      float acc[2][DOWN_ROWS_PER_LANE];
      down_rows_dot<2>(format, rows, inputs, chunks, first_chunk, acc);
      if (first_chunk == 0) {
        METAL_PRAGMA_UNROLL
        for (uint n = 0; n < 2; ++n) {
          threadgroup float* token_partials = simdgroup_partials + (route_pair[n] / top_k) * DOWN_TILE_ROWS;
          METAL_PRAGMA_UNROLL
          for (uint r = 0; r < DOWN_ROWS_PER_LANE; ++r) {
            token_partials[row_group + r * DOWN_ROW_GROUPS] += acc[n][r];
          }
        }
      }
    }
    if (index < count) {
      const uint route = route_at(index);
      const device half* const inputs[1] = {hidden + size_t(route) * hidden_dim};
      float acc[1][DOWN_ROWS_PER_LANE];
      down_rows_dot<1>(format, rows, inputs, chunks, first_chunk, acc);
      if (first_chunk == 0) {
        threadgroup float* token_partials = simdgroup_partials + (route / top_k) * DOWN_TILE_ROWS;
        METAL_PRAGMA_UNROLL
        for (uint r = 0; r < DOWN_ROWS_PER_LANE; ++r) {
          token_partials[row_group + r * DOWN_ROW_GROUPS] += acc[0][r];
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint index = thread_index; index < tile_values; index += DOWN_SIMDGROUPS * METAL_SIMD_SIZE) {
    float sum = 0.0f;
    METAL_PRAGMA_UNROLL
    for (uint group = 0; group < DOWN_SIMDGROUPS; ++group) {
      sum += partials[group * MAX_TOKENS * DOWN_TILE_ROWS + index];
    }
    const uint token = index / DOWN_TILE_ROWS;
    const uint row = index % DOWN_TILE_ROWS;
    output[size_t(token) * model_dim + tile * DOWN_TILE_ROWS + row] = T(sum);
  }
}
