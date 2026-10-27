#pragma once

#include <metal_stdlib>

#include "../../../common/thread_context.h"
#include "../../../generated/matmul.h"
#include "../generated/gemm.h"
#include "../../../hadamard_transform/hadamard_transform.h"
#include "../../../common/soft_cap.h"
#include "../../common/fragment.h"
#include "../../common/simdgroup_fragment_ops.h"
#include "../../common/quant_unpack.h"
#include "block_geometry.h"
#include "gemm_rht.h"
#include "gemm_tiling.h"
#include "operands.h"
#include "packed_w4_input.h"

using namespace metal;

namespace uzu {
namespace gemm {

template <
    typename OutputElementType_,
    GemmTiling GEMM_TILING,
    bool TRANSPOSE_B,
    typename LeftOperand,
    typename RightOperand>
struct RegisterQuantMmaCore {
  using OutputElementType = OutputElementType_;
  using Left = LeftOperand;
  using Right = RightOperand;
  using LeftElementType = typename Left::ElementType;
  using RightElementType = typename Right::ElementType;
  using LeftStorage = operands::LeftStorage<Left>;
  using RightStorage = operands::RightStorage<Right>;
  UZU_CONST bool PACKED_W4_A = UZU_GEMM_IS_PACKED_W4_TILING(GEMM_TILING);
  UZU_CONST int THREADGROUP_BLOCK_M = gemm_tiling_block_m(GEMM_TILING);
  UZU_CONST int THREADGROUP_BLOCK_N = gemm_tiling_block_n(GEMM_TILING);
  UZU_CONST int THREADGROUP_BLOCK_K = gemm_tiling_block_k(GEMM_TILING);
  UZU_CONST int SIMDGROUPS_PER_ROW = gemm_tiling_simdgroups_per_row(GEMM_TILING);
  UZU_CONST int SIMDGROUPS_PER_COLUMN = gemm_tiling_simdgroups_per_column(GEMM_TILING);
  UZU_CONST int K_SLICES = gemm_tiling_simdgroups_per_k(GEMM_TILING);
  UZU_CONST int COLS_PER_SIMDGROUP = THREADGROUP_BLOCK_N / SIMDGROUPS_PER_COLUMN;
  static_assert(K_SLICES == 4 && SIMDGROUPS_PER_COLUMN == 2 && COLS_PER_SIMDGROUP == 16);

  using RegisterGeometry = ThreadgroupTileGeometry<THREADGROUP_BLOCK_M, THREADGROUP_BLOCK_N>;
  struct RegisterTile {
    RegisterGeometry geometry;
    ushort lane;
    short2 position;
    uint partition;
    ushort k_slice;
    uint output_col;
  };

  static METAL_FUNC RegisterTile
  make_register_tile(const constant uzu::matmul::GemmParams* params, const thread ThreadContext& thread_context) {
    const ushort lane = ushort(thread_context.simd_lane_id);
    const uint2 id =
        tile_id(uint2(thread_context.threadgroup_position.x, thread_context.threadgroup_position.y), params);
    const RegisterGeometry geometry = RegisterGeometry::compute(id, params);
    const ushort simdgroup_index = ushort(thread_context.simdgroup_index);
    return {
        geometry,
        lane,
        uzu::matmul::SimdgroupFragmentOps::get_position(lane),
        thread_context.threadgroup_position.z,
        ushort(simdgroup_index / SIMDGROUPS_PER_COLUMN),
        geometry.block_col_start + uint(simdgroup_index % SIMDGROUPS_PER_COLUMN) * COLS_PER_SIMDGROUP
    };
  }

  template <ushort WORDS>
  struct W4PairDecoder {
    thread const uint* words;

    METAL_FUNC half2 operator()(const ushort row, const ushort k) const thread {
      return w4_pair(words[row * WORDS + k / 4], k % 4);
    }
  };

  template <bool TRANSPOSED, class Accumulator>
  static METAL_FUNC void finish_register_quant(
      thread Accumulator& accumulator,
      device OutputElementType* d,
      const constant uzu::matmul::GemmParams* params,
      const GemmDTransform output_transform,
      const constant float& soft_cap,
      const device int32_t* rht_factors,
      const device RightElementType* output_bias,
      threadgroup RightElementType* scratch,
      const thread RegisterTile& tile,
      const thread ThreadContext& thread_context
  ) {
    const bool output_writer = accumulator.template reduce_k_slices<K_SLICES, SIMDGROUPS_PER_COLUMN>(
        reinterpret_cast<threadgroup float*>(scratch),
        thread_context
    );
    const bool needs_rht = output_transform.contains(GemmDTransform::RHT);
    if (!output_writer && !needs_rht) {
      return;
    }

    const uint row_start = tile.geometry.block_row_start;
    const uint m_remaining = min(uint(THREADGROUP_BLOCK_M), params->M - row_start);
    const uint n_remaining =
        tile.output_col < params->N ? min(uint(COLS_PER_SIMDGROUP), params->N - tile.output_col) : 0;
    const uint safe_output_col = min(tile.output_col, params->N - 1);
    const short2 tile_dimensions =
        TRANSPOSED ? short2(short(m_remaining), short(n_remaining)) : short2(short(n_remaining), short(m_remaining));
    if (output_writer) {
      if (output_transform.contains(GemmDTransform::SCALE)) {
        const uzu::matmul::TransformScale<float, float> scale(params->ab_scale);
        accumulator.map([&](float value) { return scale.apply(value); });
      }
      const bool needs_bias = output_transform.contains(GemmDTransform::BIAS);
      if (needs_bias && !needs_rht) {
        Accumulator bias;
        constexpr int BIAS_ROW_STRIDE = TRANSPOSED ? 1 : 0;
        constexpr int BIAS_COL_STRIDE = TRANSPOSED ? 0 : 1;
        bias.load_from(
            tile.lane,
            uzu::matmul::fragment_source(output_bias + safe_output_col, BIAS_ROW_STRIDE, BIAS_COL_STRIDE)
                .bounded(tile_dimensions.y, tile_dimensions.x)
        );
        accumulator.zip_for_each_coord(
            tile.lane,
            [](short, short, thread float& value, thread float& bias_value) { value += bias_value; },
            accumulator,
            bias
        );
      }
      if (output_transform.contains(GemmDTransform::SOFT_CAP)) {
        accumulator.map([&](float value) { return uzu::apply_soft_cap(value, soft_cap); });
      }

      device OutputElementType* destination = d + size_t(tile.partition) * size_t(params->M) * size_t(params->N) +
                                              size_t(row_start) * params->leading_dimension_d + safe_output_col;
      const int row_stride = TRANSPOSED ? 1 : int(params->leading_dimension_d);
      const auto col_stride = [&]() {
        if constexpr (TRANSPOSED) {
          return int(params->leading_dimension_d);
        } else {
          return uzu::Int<1>{};
        }
      }();
      accumulator
          .store_safe(tile.lane, destination, row_stride, tile_dimensions, uzu::Int<1>{}, uzu::Int<1>{}, col_stride);
    }

    if (needs_rht) {
      const uint tile_col_start = tile.geometry.block_col_start;
      const ushort tile_block_cols = ushort(min(int(THREADGROUP_BLOCK_N), int(params->N) - int(tile_col_start)));
      device OutputElementType* tile_output = d + size_t(tile.partition) * size_t(params->M) * size_t(params->N) +
                                              size_t(row_start) * params->leading_dimension_d + tile_col_start;
      const device RightElementType* tile_bias = output_bias;
      if (output_transform.contains(GemmDTransform::BIAS)) {
        tile_bias += tile_col_start;
      }

      threadgroup_barrier(mem_flags::mem_device);
      apply_output_random_hadamard_transform(
          tile_output,
          tile_output,
          rht_factors + tile_col_start,
          tile_bias,
          output_transform.contains(GemmDTransform::BIAS),
          ushort(m_remaining),
          tile_block_cols,
          params->leading_dimension_d,
          params->leading_dimension_d,
          ushort(gemm_tiling_simdgroups_per_k(GEMM_TILING) * SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN),
          thread_context
      );
    }
  }

  static METAL_FUNC auto run_packed_w4(
      const RightStorage right,
      const device uint* packed_a,
      const constant uzu::matmul::GemmParams* params,
      const thread RegisterTile& tile
  ) {
    static_assert(Right::QUANTIZED && Right::BITS == 4, "packed route is W4 only");
    constexpr ushort GROUP_SIZE = Right::GROUP_SIZE;
    constexpr ushort STEPS = PackedW4Input<GROUP_SIZE>::STEPS;
    constexpr ushort WORDS = GROUP_SIZE / 32;
    constexpr ushort GROUP_WORDS = PackedW4Input<GROUP_SIZE>::GROUP_WORDS;
    constexpr ushort ITERATION_GROUPS = 64 / GROUP_SIZE;
    constexpr ushort TILES = 2;
    using Ops = uzu::matmul::SimdgroupFragmentOps;
    using Accumulator = uzu::matmul::Fragment<float, TILES, 1, Ops>;
    using RightFragment = uzu::matmul::Fragment<half, STEPS, 1, Ops>;
    using LeftFragment = uzu::matmul::GeneratedFragment<half, TILES, STEPS, Ops, W4PairDecoder<WORDS>>;

    const uint slot = uint(tile.position.x) / 2;
    const uint fm = uint(tile.position.y);
    const uint row_bytes = params->K / 2;
    const uint sign_mask = right.signed_codes ? W4_SIGN_MASK : 0u;

    Accumulator accumulator;
    accumulator.clear();
    for (uint iteration = 0; iteration < params->aligned_inner_iterations; ++iteration) {
      const uint block64 =
          tile.partition * params->aligned_inner_iterations * K_SLICES + iteration * K_SLICES + tile.k_slice;
      const uint first_group = block64 * ITERATION_GROUPS;
      for (ushort within_block = 0; within_block < ITERATION_GROUPS; ++within_block) {
        const uint group = first_group + within_block;
        const device uint* packed_group = packed_a + group * GROUP_WORDS;
        const device uint* input_pairs = packed_group + PackedW4Input<GROUP_SIZE>::pair_offset(slot, fm);
        uint x[STEPS];
        METAL_PRAGMA_UNROLL
        for (ushort v = 0; v < STEPS / 4; ++v) {
          const uint4 raw = reinterpret_cast<const device uint4*>(input_pairs)[v];
          x[4 * v] = raw.x;
          x[4 * v + 1] = raw.y;
          x[4 * v + 2] = raw.z;
          x[4 * v + 3] = raw.w;
        }
        const float4 sums =
            reinterpret_cast<const device float4*>(packed_group + PackedW4Input<GROUP_SIZE>::sums_offset())[slot];

        uint rows[TILES];
        float scales[TILES];
        float zero_points[TILES];
        uint words[TILES * WORDS];
        METAL_PRAGMA_UNROLL
        for (ushort t = 0; t < TILES; ++t) {
          rows[t] = min(tile.output_col + 8 * t + fm, params->N - 1);
          const device uint* source = reinterpret_cast<const device uint*>(
              right.codes + rows[t] * row_bytes + group * (GROUP_SIZE / 2) + slot * (4 * WORDS)
          );
          METAL_PRAGMA_UNROLL
          for (ushort word = 0; word < WORDS; ++word) {
            words[t * WORDS + word] = source[word] ^ sign_mask;
          }
          scales[t] = float(right.scale(rows[t], group, params));
          zero_points[t] = float(right.zero_point(rows[t], group, params));
        }

        W4PairDecoder<WORDS> decoder{words};
        LeftFragment left{decoder};
        RightFragment right_fragment;
        METAL_PRAGMA_UNROLL
        for (ushort step = 0; step < STEPS; ++step) {
          right_fragment.fragment_at(step, 0) = as_type<half2>(x[step]);
        }
        Accumulator total;
        total.clear();
        uzu::matmul::fragment_mma(total, left, right_fragment);
        accumulator.for_each_lane(
            [&](ushort t, ushort input, thread float& value, thread float& sum) {
              const float correction = fma(zero_points[t], sums[2 * input], sums[2 * input + 1]);
              value = fma(sum - correction, scales[t], value);
            },
            total
        );
      }
    }

    return accumulator;
  }

  static METAL_FUNC auto run_register_w8(
      const LeftStorage left,
      const RightStorage right,
      const constant uzu::matmul::GemmParams* params,
      const thread RegisterTile& tile
  ) {
    static_assert(Right::QUANTIZED && Right::BITS == 8 && Right::GROUP_SIZE == 64, "register route is W8 G64 only");
    constexpr ushort VALUES = 16;
    constexpr ushort TILES = 2;
    constexpr uint BLOCK = THREADGROUP_BLOCK_K;
    constexpr uint INPUT_VECTORS = VALUES / 4;
    using Ops = uzu::matmul::SimdgroupFragmentOps;
    using Accumulator = uzu::matmul::Fragment<float, 1, TILES, Ops>;

    const uint fm = uint(tile.position.y);
    const uint fn = uint(tile.position.x);
    const uint k_total = params->K;
    const uint row_bytes = k_total * Right::BITS / 8;
    const uint last_row = params->N - 1;
    const uint sign_mask = right.signed_codes ? W8_SIGN_MASK : 0u;
    const device uint8_t* weights = right.codes;

    const uint row_base = tile.output_col;
    uint rows[TILES][2];
    METAL_PRAGMA_UNROLL
    for (uint t = 0; t < TILES; ++t) {
      METAL_PRAGMA_UNROLL
      for (uint r = 0; r < 2; ++r) {
        rows[t][r] = min(row_base + 8 * t + fn + r, last_row);
      }
    }

    Accumulator accumulator;
    accumulator.clear();

    const device bfloat* a_row =
        left.values + min(tile.geometry.block_row_start + fm, params->M - 1) * k_total + VALUES * fn;
    const uint blocks = k_total / BLOCK;
    const uint blocks_per_partition = K_SLICES * params->aligned_inner_iterations + blocks % K_SLICES;
    const uint partition_block_start = tile.partition * blocks_per_partition;
    const uint partition_block_end = partition_block_start + blocks_per_partition;

    uint4 a_raw[INPUT_VECTORS];
    uint4 w[TILES][2];
    float scale[TILES][2];
    float bias[TILES][2];

    auto load_a = [&](uint block, thread uint4(&a_dst)[INPUT_VECTORS]) {
      const device uint4* source = reinterpret_cast<const device uint4*>(a_row + block * BLOCK);
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < INPUT_VECTORS; ++i) {
        a_dst[i] = source[i];
      }
    };
    auto load_weights = [&](uint block, thread uint4(&w_dst)[TILES][2]) {
      const uint k_lane = block * BLOCK + VALUES * fm;
      METAL_PRAGMA_UNROLL
      for (uint t = 0; t < TILES; ++t) {
        METAL_PRAGMA_UNROLL
        for (uint r = 0; r < 2; ++r) {
          w_dst[t][r] = *reinterpret_cast<const device uint4*>(weights + rows[t][r] * row_bytes + k_lane);
        }
      }
    };
    auto load_meta = [&](uint block, thread float (&scale_dst)[TILES][2], thread float (&bias_dst)[TILES][2]) {
      const uint group = (block * BLOCK + VALUES * fm) / Right::GROUP_SIZE;
      METAL_PRAGMA_UNROLL
      for (uint t = 0; t < TILES; ++t) {
        METAL_PRAGMA_UNROLL
        for (uint r = 0; r < 2; ++r) {
          const uint row = rows[t][r];
          scale_dst[t][r] = float(right.scale(row, group, params));
          bias_dst[t][r] = magic_dequant_bias(scale_dst[t][r]);
        }
      }
    };

    uint block = partition_block_start + tile.k_slice;
    if (block < partition_block_end) {
      load_a(block, a_raw);
      load_weights(block, w);
      load_meta(block, scale, bias);
    }
    for (; block < partition_block_end; block += 4) {
      const uint next_block = block + 4;
      const uint load_block = next_block < partition_block_end ? next_block : block;
      METAL_PRAGMA_UNROLL
      for (uint t = 0; t < TILES; ++t) {
        METAL_PRAGMA_UNROLL
        for (uint r = 0; r < 2; ++r) {
          w[t][r] ^= uint4(sign_mask);
        }
      }

      auto left_generator = [&](ushort, ushort s) -> float2 {
        const uint word0 = reinterpret_cast<const thread uint*>(a_raw)[s / 2];
        const uint word1 = reinterpret_cast<const thread uint*>(a_raw)[VALUES / 2 + s / 2];
        return as_type<float2>(
            uint2(s % 2 ? word0 & 0xffff0000u : word0 << 16, s % 2 ? word1 & 0xffff0000u : word1 << 16)
        );
      };
      auto right_generator = [&](ushort s, ushort t) -> float2 {
        const uint word_index = s * Right::BITS / 32;
        const uint word0 = w[t][0][word_index];
        const uint word1 = w[t][1][word_index];
        return float2(
            fma(magic_code_float<symmetric_zero_point<8>()>(extract_bits(word0, 8 * (s % 4), 8)),
                scale[t][0],
                bias[t][0]),
            fma(magic_code_float<symmetric_zero_point<8>()>(extract_bits(word1, 8 * (s % 4), 8)),
                scale[t][1],
                bias[t][1])
        );
      };
      using LeftFragment = uzu::matmul::GeneratedFragment<float, 1, VALUES, Ops, decltype(left_generator)>;
      using RightFragment = uzu::matmul::GeneratedFragment<float, VALUES, TILES, Ops, decltype(right_generator)>;
      LeftFragment left_fragment{left_generator};
      RightFragment right_fragment{right_generator};
      uzu::matmul::fragment_mma(accumulator, left_fragment, right_fragment);

      uint4 w_next[TILES][2];
      float scale_next[TILES][2];
      float bias_next[TILES][2];
      load_a(load_block, a_raw);
      load_meta(load_block, scale_next, bias_next);
      load_weights(load_block, w_next);
      METAL_PRAGMA_UNROLL
      for (uint t = 0; t < TILES; ++t) {
        METAL_PRAGMA_UNROLL
        for (uint r = 0; r < 2; ++r) {
          scale[t][r] = scale_next[t][r];
          bias[t][r] = bias_next[t][r];
          w[t][r] = w_next[t][r];
        }
      }
    }

    return accumulator;
  }

  static METAL_FUNC void run(
      LeftStorage left,
      RightStorage right,
      const device uint* packed_a,
      device OutputElementType* d,
      const constant uzu::matmul::GemmParams* params,
      GemmDTransform output_transform,
      const constant float& soft_cap,
      const device RightElementType* output_bias,
      const device int32_t* rht_factors,
      threadgroup RightElementType* scratch,
      const thread ThreadContext& thread_context
  ) {
    const RegisterTile tile = make_register_tile(params, thread_context);
    if (tile.geometry.out_of_bounds) {
      return;
    }

    auto accumulator = [&]() {
      if constexpr (PACKED_W4_A) {
        return run_packed_w4(right, packed_a, params, tile);
      } else {
        return run_register_w8(left, right, params, tile);
      }
    }();
    finish_register_quant<PACKED_W4_A>(
        accumulator,
        d,
        params,
        output_transform,
        soft_cap,
        rht_factors,
        output_bias,
        scratch,
        tile,
        thread_context
    );
  }
};

} // namespace gemm
} // namespace uzu
