#pragma once

#include <metal_stdlib>

#include "../common/defines.h"
#include "../generated/moe.h"
#include "../matmul/common/fragment.h"
#include "../matmul/common/mxu_fragment/ops.h"
#include "expert_common.h"

using namespace metal;

// Tiles of the grouped expert GEMMs: GROUPED_BM routes by GROUPED_BN weight rows, GROUPED_BK inputs per step.
// The eight simdgroups split the tile 2 x 4, each owning 32 routes by 32 weight rows.
#define GROUPED_BM 64
#define GROUPED_BN 128
#define GROUPED_BK 32
#define GROUPED_SIMDGROUPS 8
#define GROUPED_THREADS (GROUPED_SIMDGROUPS * METAL_SIMD_SIZE)
#define GROUPED_STRIDE (GROUPED_BK + 8)

namespace uzu {
namespace moe {

using GroupedOps = uzu::matmul::MxuFragmentOps<>;
// The matrix units multiply E inputs by E weights: bfloat for the up projection, whose bfloat activations then need
// no conversion, half for the down projection, whose activations are half.
template <typename E>
using GroupedLeft = uzu::matmul::Fragment<E, 2, 2, GroupedOps>;
template <typename E>
using GroupedRight = uzu::matmul::Fragment<E, 2, 2, GroupedOps, uzu::matmul::ReadDirect, true>;
using GroupedAccumulator = uzu::matmul::Fragment<float, 2, 2, GroupedOps>;

static_assert(GROUPED_THREADS == 4 * GROUPED_BM, "a thread stages a quarter of an input row per step");
static_assert(GROUPED_THREADS == 2 * GROUPED_BN, "a thread stages half of a weight row per step");
static_assert(GROUPED_BK == 32, "staging assumes 8 inputs and 16 weights per thread per step");

// The global data one thread stages per step, fetched a step ahead so that its latency hides behind the MMAs.
struct GroupedFetch {
  uint4 input; // 8 input values: bfloat or half
  uint2 codes; // 16 weight codes
  uint params; // NVFP4: the E4M3 scale byte; affine: the bfloat16 (scale, bias) pair
};

// Codes and quantization parameters of 16 consecutive weights of one row, starting at input `k` (a multiple of 16).
METAL_FUNC void fetch_weights16(
    const MoeExpertFormat format,
    const device uchar* row_codes,
    const device uchar* row_scales,
    const uint k,
    thread GroupedFetch& fetch
) {
  fetch.codes = *reinterpret_cast<const device uint2*>(row_codes + k / 2);
  fetch.params = format == MoeExpertFormat::Nvfp4
                     ? uint(row_scales[k / NVFP4_GROUP_SIZE])
                     : reinterpret_cast<const device uint*>(row_scales)[k / AFFINE_GROUP_SIZE];
}

// Dequantizes the 16 weights of `fetch` into `out` as E (half or bfloat): exact for NVFP4, whose E2M1 times E4M3
// products need at most six significant bits; affine weights round as the bfloat16 reference dequantization does.
template <typename E>
METAL_FUNC void store_weights16(const MoeExpertFormat format, const thread GroupedFetch& fetch, threadgroup E* out) {
  using E4 = vec<E, 4>;
  float scale;
  float bias = 0.0f;
  if (format == MoeExpertFormat::Nvfp4) {
    // decode_e4m3 and decode_e2m1x8 each scale their value by a power of two; undo both in float.
    scale = decode_e4m3(fetch.params) * NVFP4_DECODE_SCALE;
  } else {
    scale = as_type<float>(fetch.params << 16);
    bias = as_type<float>(fetch.params & 0xFFFF0000u);
  }
  METAL_PRAGMA_UNROLL
  for (uint word = 0; word < 2; ++word) {
    E4 low;
    E4 high;
    if (format == MoeExpertFormat::Nvfp4) {
      half2 pairs[4];
      decode_e2m1x8(fetch.codes[word], pairs);
      float2 values[4];
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < 4; ++i) {
        values[i] = float2(pairs[i]) * scale;
      }
      low = E4(float4(values[0].x, values[1].x, values[2].x, values[3].x));
      high = E4(float4(values[0].y, values[1].y, values[2].y, values[3].y));
    } else {
      half2 even[2];
      half2 odd[2];
      decode_u4x8(fetch.codes[word], even, odd);
      // even holds (q0, q4), (q2, q6); odd holds 16 times (q1, q5), (q3, q7).
      const float odd_scale = scale * 0.0625f;
      const float2 even0 = fma(float2(even[0]), scale, bias);
      const float2 even1 = fma(float2(even[1]), scale, bias);
      const float2 odd0 = fma(float2(odd[0]), odd_scale, bias);
      const float2 odd1 = fma(float2(odd[1]), odd_scale, bias);
      low = E4(float4(even0.x, odd0.x, even1.x, odd1.x));
      high = E4(float4(even0.y, odd0.y, even1.y, odd1.y));
    }
    *reinterpret_cast<threadgroup E4*>(out + 8 * word) = low;
    *reinterpret_cast<threadgroup E4*>(out + 8 * word + 4) = high;
  }
}

// Runs the steps of one tile: `fetch(k)` loads what this thread stages for inputs [k, k + GROUPED_BK), `commit`
// writes it to the input and weight tiles. The next step's fetch is issued before this step's MMAs.
template <typename E, typename Fetch, typename Commit>
METAL_FUNC void grouped_tile_mma(
    Fetch fetch,
    Commit commit,
    const uint depth,
    threadgroup E* input_tile,
    threadgroup E* weight_tile,
    const uint simdgroup_m,
    const uint simdgroup_n,
    const ushort lane,
    thread GroupedAccumulator& accumulator
) {
  accumulator.clear();
  const auto left_source =
      uzu::matmul::fragment_source(input_tile + simdgroup_m * 32 * GROUPED_STRIDE, int(GROUPED_STRIDE));
  const auto right_source =
      uzu::matmul::fragment_source(weight_tile + simdgroup_n * 32 * GROUPED_STRIDE, int(GROUPED_STRIDE));
  GroupedFetch staged = fetch(0u);
  for (uint k = 0; k < depth; k += GROUPED_BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    commit(staged);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k + GROUPED_BK < depth) {
      staged = fetch(k + GROUPED_BK);
    }
    GroupedLeft<E> left;
    GroupedRight<E> right;
    left.load_from(lane, left_source);
    right.load_from(lane, right_source);
    uzu::matmul::fragment_mma(accumulator, left, right);
  }
}

// Calls `fn(row, column, values...)` for the accumulator elements of this lane, `row` and `column` within the
// simdgroup's 32 x 32 block and `values` one element per fragment column pair: element (row, column) of column
// fragment 0 and of column fragment 1 for `column` in [0, 16).
template <typename Fn>
METAL_FUNC void for_each_column_pair(thread GroupedAccumulator& accumulator, const ushort lane, Fn fn) {
  const short2 position = GroupedOps::get_position(lane);
  thread float* data = accumulator.elements();
  METAL_PRAGMA_UNROLL
  for (ushort row_fragment = 0; row_fragment < 2; ++row_fragment) {
    METAL_PRAGMA_UNROLL
    for (ushort element = 0; element < GroupedOps::ELEMENTS_PER_THREAD; ++element) {
      const short2 offset = GroupedOps::get_element_offset(element);
      const ushort row = row_fragment * GroupedOps::FRAGMENT_ROWS + position.y + offset.y;
      const ushort column = position.x + offset.x;
      const thread float& first = data[(row_fragment * 2) * GroupedOps::ELEMENTS_PER_THREAD + element];
      const thread float& second = data[(row_fragment * 2 + 1) * GroupedOps::ELEMENTS_PER_THREAD + element];
      fn(row, column, first, second);
    }
  }
}

} // namespace moe
} // namespace uzu
