#pragma once

#include <metal_stdlib>

#include "../../../../common/thread_context.h"
#include "../../../../generated/gemm.h"
#include "../../../../generated/matmul.h"
#include "../../../common/fragment.h"
#include "../../../common/mxu_fragment/ops.h"
#include "tile_context.h"

using namespace metal;

namespace uzu {
namespace gemm {
namespace schedules {

template <typename LeftOperand, typename RightOperand>
struct SymRow2Schedule {
  template <typename Core, bool ALIGNED_M, bool ALIGNED_N>
  static METAL_FUNC typename Core::AccumFragment launch(
      typename Core::LeftStorage left,
      typename Core::RightStorage right,
      threadgroup typename Core::RightElementType*,
      const constant uzu::matmul::GemmParams* params,
      const TileContext tile,
      const GemmAlignment,
      const thread ThreadContext& thread_context
  ) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    static_assert(LeftOperand::QUANTIZED && RightOperand::BITS == 2, "SymRow2 needs int8 x 2-bit inputs");
    static_assert(RightOperand::GROUP_SIZE == 128, "SymRow2 uses 128-column groups");
    static_assert(Core::SIMDGROUPS_PER_ROW == 1 && Core::SIMDGROUP_BLOCK_M == 16, "SymRow2 uses 16-row tiles");
    static_assert(Core::SIMDGROUP_BLOCK_N == 16 && Core::TILES_N == 1, "one SIMDgroup owns one 16-column tile");
    static_assert(Core::SIMDGROUPS_PER_K == 2 || Core::SIMDGROUPS_PER_K == 4, "SymRow2 K split is 2 or 4");

    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        16,
        16,
        32,
        false,
        true,
        false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
    );
    mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> op;
    auto a_co = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
    auto b_co = op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
    auto cT = op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(a_co)>,
        metal::remove_addrspace_t<decltype(b_co)>,
        int32_t>();

    const ushort lane = ushort(thread_context.simd_lane_id);
    const short2 position = Core::FragmentOps::get_position(lane);
    const int mrow = int(position.y);
    const int nb = int(position.x);
    const int q = int(lane & 1u) + 2 * int((lane >> 3) & 1u);
    const uint live_rows = uint(max(tile.simdgroup_limit_m, short(1)));
    const uint mr0 = tile.abs_row_base + uint(min(mrow, int(live_rows) - 1));
    const uint mr1 = tile.abs_row_base + uint(min(mrow + 8, int(live_rows) - 1));
    const uint kgroups = params->K / 128;
    const device uint4* packed =
        reinterpret_cast<const device uint4*>(right.codes) + size_t(tile.absolute_column_base() / 16) * 32 + lane;
    const device int8_t* a0 = left.codes + size_t(mr0) * params->leading_dimension_a + 4 * q;
    const device int8_t* a1 = left.codes + size_t(mr1) * params->leading_dimension_a + 4 * q;

    typename Core::AccumFragment accumulator;
    accumulator.clear();
    bool mapping_ok = int(cT.get_capacity()) == 8 && int(b_co.get_capacity()) == 16 && int(a_co.get_capacity()) == 16;
    METAL_PRAGMA_UNROLL
    for (int element = 0; element < 8; ++element) {
      auto ids = cT.get_multidimensional_index(element);
      mapping_ok = mapping_ok && int(ids[1]) == mrow + 8 * ((element >> 2) & 1) && int(ids[0]) == nb + (element & 3);
    }
    METAL_PRAGMA_UNROLL
    for (int element = 0; element < 16; ++element) {
      auto b_ids = b_co.get_multidimensional_index(element);
      auto a_ids = a_co.get_multidimensional_index(element);
      mapping_ok = mapping_ok && int(b_ids[0]) == 16 * (element >> 3) + 4 * q + (element & 3);
      mapping_ok = mapping_ok && int(b_ids[1]) == mrow + 8 * ((element >> 2) & 1);
      mapping_ok = mapping_ok && int(a_ids[0]) == 16 * (element >> 3) + 4 * q + (element & 3);
      mapping_ok = mapping_ok && int(a_ids[1]) == mrow + 8 * ((element >> 2) & 1);
    }

    const uint first_k_slice =
        uint(thread_context.simdgroup_index) / uint(Core::SIMDGROUPS_PER_ROW * Core::SIMDGROUPS_PER_COLUMN);
    // The next group's loads are issued one iteration ahead to overlap the MMAs.
    uint4 weight = uint4(0);
    float scale0 = 0.0f, scale1 = 0.0f, correction0 = 0.0f, correction1 = 0.0f;
    // Clamped to the last group instead of branching; a clamped load is never consumed.
    const uint last_group = kgroups - 1;
    {
      const uint first = min(first_k_slice, last_group);
      weight = packed[first * params->leading_dimension_b];
      scale0 = left.scales[mr0 * kgroups + first];
      scale1 = left.scales[mr1 * kgroups + first];
      correction0 = scale0 * float(left.group_sums[mr0 * kgroups + first]);
      correction1 = scale1 * float(left.group_sums[mr1 * kgroups + first]);
    }
    for (uint g = first_k_slice; g < kgroups; g += Core::SIMDGROUPS_PER_K) {
      const uint4 current_weight = weight;
      const float current_scale0 = scale0, current_scale1 = scale1;
      const float current_correction0 = correction0, current_correction1 = correction1;
      const uint next = min(g + Core::SIMDGROUPS_PER_K, last_group);
      {
        weight = packed[next * params->leading_dimension_b];
        scale0 = left.scales[mr0 * kgroups + next];
        scale1 = left.scales[mr1 * kgroups + next];
        correction0 = scale0 * float(left.group_sums[mr0 * kgroups + next]);
        correction1 = scale1 * float(left.group_sums[mr1 * kgroups + next]);
      }

      METAL_PRAGMA_UNROLL
      for (int element = 0; element < 8; ++element) {
        cT[element] = 0;
      }
      METAL_PRAGMA_UNROLL
      for (int u = 0; u < 4; ++u) {
        const uint packed_word = u == 0   ? current_weight.x
                                 : u == 1 ? current_weight.y
                                 : u == 2 ? current_weight.z
                                          : current_weight.w;
        METAL_PRAGMA_UNROLL
        for (int s = 0; s < 4; ++s) {
          const uint codes = (packed_word >> (2 * s)) & 0x03030303u;
          METAL_PRAGMA_UNROLL
          for (int i = 0; i < 4; ++i) {
            b_co[s * 4 + i] = int8_t(codes >> (8 * i));
          }
        }
        METAL_PRAGMA_UNROLL
        for (int j = 0; j < 2; ++j) {
          const uint x0 = *reinterpret_cast<const device uint*>(a0 + g * 128 + 32 * u + 16 * j);
          const uint x1 = *reinterpret_cast<const device uint*>(a1 + g * 128 + 32 * u + 16 * j);
          METAL_PRAGMA_UNROLL
          for (int i = 0; i < 4; ++i) {
            a_co[8 * j + i] = int8_t(x0 >> (8 * i));
            a_co[8 * j + 4 + i] = int8_t(x1 >> (8 * i));
          }
        }
        op.run(a_co, b_co, cT);
      }

      METAL_PRAGMA_UNROLL
      for (int element = 0; element < 4; ++element) {
        accumulator.elements()[element] =
            fma(-1.5f, current_correction0, fma(current_scale0, float(cT[element]), accumulator.elements()[element]));
        accumulator.elements()[4 + element] =
            fma(-1.5f,
                current_correction1,
                fma(current_scale1, float(cT[4 + element]), accumulator.elements()[4 + element]));
      }
    }

    if (!simd_all(mapping_ok)) {
      accumulator.map([](auto) { return as_type<float>(0x7fc00000u); });
    }
    return accumulator;
  }

  template <typename Fragment, ushort K_SPLIT>
  static METAL_FUNC void reduce_k_partials(
      thread Fragment& accumulator,
      threadgroup float* partials,
      const ushort k_slice,
      const ushort tile_index,
      const ushort lane
  ) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    static_assert(K_SPLIT == 2 || K_SPLIT == 4, "SymRow2 K split is 2 or 4");
    static_assert(Fragment::ELEMENTS_PER_FRAGMENT == 8, "SymRow2 uses eight accumulators per lane");
    thread float* values = accumulator.elements();
    threadgroup float4* vector_partials = reinterpret_cast<threadgroup float4*>(partials);
    if (k_slice != 0) {
      METAL_PRAGMA_UNROLL
      for (ushort element = 0; element < Fragment::ELEMENTS_PER_FRAGMENT; element += 4) {
        const uint partial =
            ((uint(tile_index) * uint(Fragment::ELEMENTS_PER_FRAGMENT / 4) + uint(element / 4)) * uint(K_SPLIT - 1) +
             uint(k_slice - 1)) *
                METAL_SIMD_SIZE +
            lane;
        vector_partials[partial] = *reinterpret_cast<thread float4*>(values + element);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k_slice == 0) {
      METAL_PRAGMA_UNROLL
      for (ushort element = 0; element < Fragment::ELEMENTS_PER_FRAGMENT; element += 4) {
        const uint partial = (uint(tile_index) * uint(Fragment::ELEMENTS_PER_FRAGMENT / 4) + uint(element / 4)) *
                                 uint(K_SPLIT - 1) * METAL_SIMD_SIZE +
                             lane;
        thread float4& value = *reinterpret_cast<thread float4*>(values + element);
        const float4 slice1 = vector_partials[partial + 0 * METAL_SIMD_SIZE];
        if constexpr (K_SPLIT == 2) {
          value += slice1;
        } else {
          const float4 slice2 = vector_partials[partial + 1 * METAL_SIMD_SIZE];
          const float4 slice3 = vector_partials[partial + 2 * METAL_SIMD_SIZE];
          value = (value + slice1) + (slice2 + slice3);
        }
      }
    }
  }

  template <typename Fragment, typename RightStorage>
  static METAL_FUNC void apply_row_scale(
      thread Fragment& accumulator,
      const RightStorage right,
      const uint output_column,
      const ushort lane
  ) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    const uint column = output_column + uint(uzu::matmul::MxuFragmentOps<>::get_position(lane).x);
    thread float* values = accumulator.elements();
    METAL_PRAGMA_UNROLL
    for (ushort element = 0; element < 4; ++element) {
      const float scale = float(right.scales[column + element]);
      values[element] = values[element] * scale;
      values[4 + element] = values[4 + element] * scale;
    }
  }
};

} // namespace schedules
} // namespace gemm
} // namespace uzu
