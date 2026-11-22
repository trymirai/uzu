#pragma once

#include <metal_stdlib>

#include "../common/defines.h"
#include "../generated/moe.h"
#include "../matmul/common/fragment.h"
#include "../matmul/common/mxu_fragment/ops.h"
#include "expert_common.h"

using namespace metal;

// Tiles of the grouped GEMMs over affine 4-bit experts. A threadgroup holds GROUPED_AFFINE_BN weight rows of one
// expert for GROUPED_AFFINE_BK inputs, dequantized once per step, and runs them against all of its routes in
// m-tiles of GROUPED_AFFINE_MT, up to GROUPED_AFFINE_MTILES at a time: an expert pads its routes to a multiple of
// 16 rather than of a whole pass. Simdgroup (n, m) owns weight rows [32 n, 32 n + 32) and the m-tiles of parity m.
#define GROUPED_AFFINE_BN 128
#define GROUPED_AFFINE_BK 32
#define GROUPED_AFFINE_MT 16
#define GROUPED_AFFINE_MTILES 4
#define GROUPED_AFFINE_PASS_ROUTES (GROUPED_AFFINE_MT * GROUPED_AFFINE_MTILES)
#define GROUPED_AFFINE_SIMDGROUPS 8
#define GROUPED_AFFINE_SIMDGROUPS_N (GROUPED_AFFINE_BN / 32)
#define GROUPED_AFFINE_SIMDGROUPS_M (GROUPED_AFFINE_SIMDGROUPS / GROUPED_AFFINE_SIMDGROUPS_N)
#define GROUPED_AFFINE_MTILES_PER_SIMDGROUP (GROUPED_AFFINE_MTILES / GROUPED_AFFINE_SIMDGROUPS_M)
#define GROUPED_AFFINE_THREADS (GROUPED_AFFINE_SIMDGROUPS * METAL_SIMD_SIZE)
#define GROUPED_AFFINE_STRIDE (GROUPED_AFFINE_BK + 8)

namespace uzu {
namespace moe {

using GroupedAffineOps = uzu::matmul::MxuFragmentOps<>;
template <typename E>
using GroupedAffineLeft = uzu::matmul::Fragment<E, 1, 2, GroupedAffineOps>;
template <typename E>
using GroupedAffineRight = uzu::matmul::Fragment<E, 2, 2, GroupedAffineOps, uzu::matmul::ReadDirect, true>;
using GroupedAffineAccumulator = uzu::matmul::Fragment<float, 1, 2, GroupedAffineOps>;

static_assert(GROUPED_AFFINE_THREADS == 2 * GROUPED_AFFINE_BN, "a thread stages half of a weight row per step");
static_assert(GROUPED_AFFINE_THREADS == 4 * GROUPED_AFFINE_PASS_ROUTES, "a thread stages a quarter input row");
static_assert(GROUPED_AFFINE_MTILES % GROUPED_AFFINE_SIMDGROUPS_M == 0, "simdgroup rows share the m-tiles evenly");
static_assert(GROUPED_AFFINE_BK == CHUNK_VALUES, "a weight row step is one 16-byte chunk of codes");

// The global data one thread stages per step, fetched a step ahead so that its latency hides behind the MMAs.
struct GroupedAffineFetch {
  uint4 input; // 8 input values: bfloat or half
  uint2 codes; // 16 weight codes
  uint params; // the bfloat16 (scale, bias) pair of their group
};

// Dequantizes the 16 weights of `fetch` into `out` as E (half or bfloat), rounding as the bfloat16 reference
// dequantization does.
template <typename E>
METAL_FUNC void store_affine_weights16(const thread GroupedAffineFetch& fetch, threadgroup E* out) {
  using E4 = vec<E, 4>;
  const float scale = as_type<float>(fetch.params << 16);
  const float bias = as_type<float>(fetch.params & 0xFFFF0000u);
  // decode_u4x8's odd pairs hold sixteen times their codes.
  const float odd_scale = scale * 0.0625f;
  METAL_PRAGMA_UNROLL
  for (uint word = 0; word < 2; ++word) {
    half2 even[2];
    half2 odd[2];
    decode_u4x8(fetch.codes[word], even, odd);
    // even holds (q0, q4), (q2, q6); odd holds 16 times (q1, q5), (q3, q7).
    const float2 even0 = fma(float2(even[0]), scale, bias);
    const float2 even1 = fma(float2(even[1]), scale, bias);
    const float2 odd0 = fma(float2(odd[0]), odd_scale, bias);
    const float2 odd1 = fma(float2(odd[1]), odd_scale, bias);
    *reinterpret_cast<threadgroup E4*>(out + 8 * word) = E4(float4(even0.x, odd0.x, even1.x, odd1.x));
    *reinterpret_cast<threadgroup E4*>(out + 8 * word + 4) = E4(float4(even0.y, odd0.y, even1.y, odd1.y));
  }
}

// One pass over the routes [first, first + rows) of a threadgroup's expert: every step stages the threadgroup's
// weight rows once and runs them against each of the pass's m-tiles. `fetch(k)` loads what this thread stages for
// inputs [k, k + GROUPED_AFFINE_BK) and `commit` writes it to the tiles; the next step's fetch is issued before
// this step's MMAs. Accumulator j covers routes [16 i, 16 i + 16) of the pass, i = simdgroup_m + j * SIMDGROUPS_M.
template <typename E, typename Fetch, typename Commit>
METAL_FUNC void grouped_affine_pass(
    Fetch fetch,
    Commit commit,
    const uint depth,
    const uint mtiles,
    threadgroup E* input_tile,
    threadgroup E* weight_tile,
    const uint simdgroup_n,
    const uint simdgroup_m,
    const ushort lane,
    thread GroupedAffineAccumulator (&accumulators)[GROUPED_AFFINE_MTILES_PER_SIMDGROUP]
) {
  METAL_PRAGMA_UNROLL
  for (uint j = 0; j < GROUPED_AFFINE_MTILES_PER_SIMDGROUP; ++j) {
    accumulators[j].clear();
  }
  const auto right_source =
      uzu::matmul::fragment_source(weight_tile + simdgroup_n * 32 * GROUPED_AFFINE_STRIDE, int(GROUPED_AFFINE_STRIDE));
  GroupedAffineFetch staged = fetch(0u);
  for (uint k = 0; k < depth; k += GROUPED_AFFINE_BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    commit(staged);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k + GROUPED_AFFINE_BK < depth) {
      staged = fetch(k + GROUPED_AFFINE_BK);
    }
    GroupedAffineRight<E> right;
    right.load_from(lane, right_source);
    METAL_PRAGMA_UNROLL
    for (uint j = 0; j < GROUPED_AFFINE_MTILES_PER_SIMDGROUP; ++j) {
      const uint i = simdgroup_m + j * GROUPED_AFFINE_SIMDGROUPS_M;
      if (i < mtiles) {
        GroupedAffineLeft<E> left;
        left.load_from(
            lane,
            uzu::matmul::fragment_source(
                input_tile + i * GROUPED_AFFINE_MT * GROUPED_AFFINE_STRIDE,
                int(GROUPED_AFFINE_STRIDE)
            )
        );
        uzu::matmul::fragment_mma(accumulators[j], left, right);
      }
    }
  }
}

// Calls `fn(row, column, first, second)` for this lane's elements of an accumulator: `row` within its 16 routes,
// `column` in [0, 16), `first` the element of weight row `column` of the simdgroup's 32 and `second` of row
// 16 + `column`.
template <typename Fn>
METAL_FUNC void for_each_affine_pair(thread GroupedAffineAccumulator& accumulator, const ushort lane, Fn fn) {
  const short2 position = GroupedAffineOps::get_position(lane);
  thread float* data = accumulator.elements();
  METAL_PRAGMA_UNROLL
  for (ushort element = 0; element < GroupedAffineOps::ELEMENTS_PER_THREAD; ++element) {
    const short2 offset = GroupedAffineOps::get_element_offset(element);
    fn(ushort(position.y + offset.y),
       ushort(position.x + offset.x),
       data[element],
       data[GroupedAffineOps::ELEMENTS_PER_THREAD + element]);
  }
}

} // namespace moe
} // namespace uzu
