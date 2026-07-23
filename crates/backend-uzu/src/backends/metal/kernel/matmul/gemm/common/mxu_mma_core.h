#pragma once

#include "../../../common/integral_constant.h"
#include "../../../common/thread_context.h"
#include "../../../hadamard_transform/hadamard_transform.h"
#include "../../common/fragment.h"
#include "gemm_rht.h"
#include "../../common/mxu_fragment_ops.h"
#include "../../common/mxu_gemm_loop.h"
#include "../../../generated/matmul.h"
#include "../generated/gemm.h"
#include "block_geometry.h"
#include "gemm_tiling.h"
#include "quant_pack.h"
#include "quant_scale_bias.h"
#include "quant_scale_zero_point.h"

using namespace metal;

namespace uzu {
namespace gemm {

template <
    typename AT,
    typename BT,
    typename DT,
    GemmTiling GEMM_TILING,
    bool TRANSPOSE_B,
    GemmBPrologueKind B_PROLOGUE = GemmBPrologueKind::FullPrecision,
    int BITS = 0,
    int GROUP_SIZE = 0,
    GemmAPrologueKind A_PROLOGUE = GemmAPrologueKind::FullPrecision>
struct MxuMmaCore {
  UZU_CONST ushort THREADGROUP_BLOCK_M = gemm_tiling_block_m(GEMM_TILING);
  UZU_CONST ushort THREADGROUP_BLOCK_N = gemm_tiling_block_n(GEMM_TILING);
  UZU_CONST ushort SIMDGROUPS_PER_ROW = gemm_tiling_simdgroups_per_row(GEMM_TILING);
  UZU_CONST ushort SIMDGROUPS_PER_COLUMN = gemm_tiling_simdgroups_per_column(GEMM_TILING);
  UZU_CONST ushort SIMDGROUP_BLOCK_M = THREADGROUP_BLOCK_M / SIMDGROUPS_PER_ROW;
  UZU_CONST ushort SIMDGROUP_BLOCK_N = THREADGROUP_BLOCK_N / SIMDGROUPS_PER_COLUMN;
  UZU_CONST ushort SIMDGROUP_BLOCK_K = static_cast<ushort>(MXU_SIMDGROUP_BLOCK_K);
  UZU_CONST ushort THREADGROUP_BLOCK_K_FP = gemm_tiling_block_k(GEMM_TILING);
  static_assert(
      THREADGROUP_BLOCK_K_FP % SIMDGROUP_BLOCK_K == 0,
      "FP THREADGROUP_BLOCK_K must be a multiple of SIMDGROUP_BLOCK_K"
  );
  UZU_CONST ushort TILES_M = SIMDGROUP_BLOCK_M / uzu::matmul::MxuFragmentOps<>::FRAGMENT_ROWS;
  UZU_CONST ushort TILES_N = SIMDGROUP_BLOCK_N / uzu::matmul::MxuFragmentOps<>::FRAGMENT_COLS;
  UZU_CONST ushort TILES_K = SIMDGROUP_BLOCK_K / uzu::matmul::MxuFragmentOps<>::FRAGMENT_ROWS;

  UZU_CONST ushort QUANT_BK = (B_PROLOGUE == GemmBPrologueKind::FullPrecision) ? 0 : GROUP_SIZE;
  UZU_CONST ushort PADDING_B = 16 / sizeof(BT);
  UZU_CONST ushort SHARED_STRIDE_B = (QUANT_BK > 0) ? (QUANT_BK + PADDING_B) : 1;
  UZU_CONST ushort THREADGROUP_THREADS = SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN * METAL_SIMD_SIZE;
  static_assert(
      B_PROLOGUE == GemmBPrologueKind::FullPrecision || QUANT_BK % SIMDGROUP_BLOCK_K == 0,
      "QUANT_BK must be a multiple of SIMDGROUP_BLOCK_K"
  );
  static_assert(
      B_PROLOGUE == GemmBPrologueKind::FullPrecision || THREADGROUP_BLOCK_K_FP % QUANT_BK == 0,
      "Tile block_k must be a multiple of QUANT_BK"
  );

  using AccumulatorType = float;

  using BLoaderScaleBias = QuantizedBlockLoaderScaleBias<
      BT,
      THREADGROUP_BLOCK_N,
      (QUANT_BK > 0) ? QUANT_BK : 1,
      SHARED_STRIDE_B,
      1,
      THREADGROUP_THREADS,
      (GROUP_SIZE > 0) ? GROUP_SIZE : 1,
      (BITS > 0) ? BITS : 4>;
  using BLoaderScaleZeroPoint = QuantizedBlockLoaderScaleZeroPoint<
      BT,
      THREADGROUP_BLOCK_N,
      (QUANT_BK > 0) ? QUANT_BK : 1,
      SHARED_STRIDE_B,
      1,
      THREADGROUP_THREADS,
      (GROUP_SIZE > 0) ? GROUP_SIZE : 1,
      (BITS > 0) ? BITS : 4>;
  using BLoaderScaleSymmetric = QuantizedBlockLoaderScaleZeroPoint<
      BT,
      THREADGROUP_BLOCK_N,
      (QUANT_BK > 0) ? QUANT_BK : 1,
      SHARED_STRIDE_B,
      1,
      THREADGROUP_THREADS,
      (GROUP_SIZE > 0) ? GROUP_SIZE : 1,
      (BITS > 0) ? BITS : 4,
      true>;

  using AccumFragment = uzu::matmul::Fragment<AccumulatorType, TILES_M, TILES_N, uzu::matmul::MxuFragmentOps<>>;

  struct QuantizedWeightAddressing {
    int row_stride_bytes;
    int groups_per_row;
    int k_offset_groups;
    const device uint8_t* block;
  };

  static METAL_FUNC QuantizedWeightAddressing
  quantized_weight_addressing(const device BT* b, const size_t block_col, const uint k_offset, const int k_elements) {
    constexpr int pack_factor = get_pack_factor<(BITS > 0) ? BITS : 4, 8>();
    constexpr int bytes_per_pack = get_bytes_per_pack<(BITS > 0) ? BITS : 4>();
    const int row_stride_bytes = k_elements * bytes_per_pack / pack_factor;
    return QuantizedWeightAddressing{
        row_stride_bytes,
        (k_elements + int(GROUP_SIZE) - 1) / int(GROUP_SIZE),
        int(k_offset) / int(GROUP_SIZE),
        reinterpret_cast<const device uint8_t*>(b) + block_col * row_stride_bytes +
            int(k_offset) * bytes_per_pack / pack_factor,
    };
  }

  template <bool ALIGNED_M, bool ALIGNED_N, typename Loader>
  static METAL_FUNC AccumFragment quant_k_loop(
      const device AT* a_simdgroup,
      threadgroup BT* b_shared,
      const int leading_dimension_a,
      const int aligned_k_iterations,
      const short simdgroup_limit_m,
      const short simdgroup_limit_n,
      const ushort tile_col_offset,
      const ushort tile_block_cols,
      thread Loader& loader_b,
      const thread ThreadContext& thread_context
  ) {
    AccumFragment accumulator;
    accumulator.clear();

    threadgroup BT* b_shared_simdgroup = b_shared + tile_col_offset * SHARED_STRIDE_B;
    const short2 tile_dimensions_b = short2(QUANT_BK, tile_block_cols);

    METAL_PRAGMA_NO_UNROLL
    for (int outer_k = 0; outer_k < aligned_k_iterations; ++outer_k) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if constexpr (ALIGNED_N) {
        loader_b.load_unsafe();
      } else {
        loader_b.load_safe(tile_dimensions_b);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      METAL_PRAGMA_NO_UNROLL
      for (int inner_k = 0; inner_k < QUANT_BK; inner_k += SIMDGROUP_BLOCK_K) {
        uzu::matmul::Fragment<AT, TILES_M, TILES_K, uzu::matmul::MxuFragmentOps<>> left_tile;
        uzu::matmul::Fragment<BT, TILES_N, TILES_K, uzu::matmul::MxuFragmentOps<>> right_tile;

        const int left_offset = inner_k;
        auto left_src = uzu::matmul::fragment_source(a_simdgroup + left_offset, leading_dimension_a);
        if constexpr (!ALIGNED_M) {
          left_src = left_src.bounded(simdgroup_limit_m, SIMDGROUP_BLOCK_K);
        }
        left_tile.load_from(thread_context.simd_lane_id, left_src);

        right_tile.load_from(
            thread_context.simd_lane_id,
            uzu::matmul::fragment_source(b_shared_simdgroup + inner_k, int(SHARED_STRIDE_B))
        );

        uzu::matmul::MxuFragmentOps<>::template fragment_mma<false, true>(accumulator, left_tile, right_tile);
      }

      a_simdgroup += QUANT_BK;
      loader_b.next();
    }

    return accumulator;
  }

  template <bool ALIGNED_M>
  static METAL_FUNC uzu::matmul::Fragment<int8_t, TILES_M, TILES_K, uzu::matmul::MxuFragmentOps<>> load_int8_left_tile(
      const device int8_t* a_int8_simdgroup,
      const int leading_dimension_a,
      const short simdgroup_limit_m,
      const ushort simd_lane_id
  ) {
    uzu::matmul::Fragment<int8_t, TILES_M, TILES_K, uzu::matmul::MxuFragmentOps<>> left_tile;
    auto left_src = uzu::matmul::fragment_source(a_int8_simdgroup, leading_dimension_a);
    if constexpr (!ALIGNED_M) {
      left_src = left_src.bounded(simdgroup_limit_m, SIMDGROUP_BLOCK_K);
    }
    left_tile.load_from(simd_lane_id, left_src);
    return left_tile;
  }

  template <bool ALIGNED_M, typename Ops>
  static METAL_FUNC void fill_row_group_cache(
      thread float* cache,
      const device float* scales,
      const short2 position,
      const short simdgroup_limit_m,
      const uint abs_row_base,
      const uint groups_per_row,
      const uint group_index
  ) {
    METAL_PRAGMA_UNROLL
    for (ushort tile_row = 0; tile_row < TILES_M; ++tile_row) {
      METAL_PRAGMA_UNROLL
      for (ushort thread_row = 0; thread_row < Ops::THREAD_ELEMENT_ROWS; ++thread_row) {
        const short row = position.y + tile_row * Ops::FRAGMENT_ROWS + thread_row * Ops::THREAD_ELEMENT_ROW_STRIDE;
        const ushort cache_index = tile_row * Ops::THREAD_ELEMENT_ROWS + thread_row;
        if (ALIGNED_M || row < simdgroup_limit_m) {
          cache[cache_index] = scales[(abs_row_base + uint(row)) * groups_per_row + group_index];
        }
      }
    }
  }

  template <bool ALIGNED_N, typename Ops, typename ValueForGroup>
  static METAL_FUNC void fill_column_group_cache(
      thread float* cache,
      const short2 position,
      const short simdgroup_limit_n,
      const uint abs_col_base,
      const uint groups_per_row,
      const uint group_index,
      ValueForGroup value_for_group
  ) {
    METAL_PRAGMA_UNROLL
    for (ushort tile_col = 0; tile_col < TILES_N; ++tile_col) {
      METAL_PRAGMA_UNROLL
      for (ushort thread_col = 0; thread_col < Ops::THREAD_ELEMENT_COLS; ++thread_col) {
        const short col = position.x + tile_col * Ops::FRAGMENT_COLS + thread_col;
        const ushort cache_index = tile_col * Ops::THREAD_ELEMENT_COLS + thread_col;
        if (ALIGNED_N || col < simdgroup_limit_n) {
          cache[cache_index] = value_for_group((abs_col_base + uint(col)) * groups_per_row + group_index);
        }
      }
    }
  }

  static_assert(
      A_PROLOGUE != GemmAPrologueKind::Int8Symmetric || GROUP_SIZE % int(SIMDGROUP_BLOCK_K) == 0,
      "A8 weight group size must be a multiple of the 32-wide activation chunk"
  );
  UZU_CONST bool int8_activation_needs_weight_correction =
      A_PROLOGUE == GemmAPrologueKind::Int8Symmetric && B_PROLOGUE != GemmBPrologueKind::ScaleSymmetricDequant;
  UZU_CONST float int8_weight_midpoint = (BITS == 4) ? 8.0f : 128.0f;

  static METAL_FUNC float int8_weight_correction_coefficient(
      const uint col,
      const uint weight_group,
      const uint weight_groups_per_row,
      const device BT* b_scales,
      const device BT* biases,
      const device uint8_t* zero_points
  ) {
    const uint scale_index = col * weight_groups_per_row + weight_group;
    const float scale = static_cast<float>(b_scales[scale_index]);
    if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant) {
      return static_cast<float>(biases[scale_index]) + scale * int8_weight_midpoint;
    } else if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant) {
      float zero_point;
      if constexpr (BITS == 4) {
        const uint zp_row_stride = (weight_groups_per_row + 1u) / 2u;
        const uchar packed = zero_points[col * zp_row_stride + (weight_group / 2u)];
        zero_point = float((weight_group & 1u) == 0u ? (packed & 0x0fu) : (packed >> 4));
      } else {
        zero_point = float(zero_points[scale_index]);
      }
      return scale * (int8_weight_midpoint - zero_point);
    } else {
      return 0.0f;
    }
  }

  template <bool ALIGNED_M, bool ALIGNED_N>
  static METAL_FUNC AccumFragment symmetric_int8_activation_k_loop(
      const device int8_t* a_int8_simdgroup,
      const device uint8_t* b_packed_simdgroup,
      const device float* a_scales,
      const device BT* b_scales,
      const device BT* biases,
      const device uint8_t* zero_points,
      const int leading_dimension_a,
      const int b_row_stride_bytes,
      const int weight_group_iterations,
      const short simdgroup_limit_m,
      const short simdgroup_limit_n,
      const uint abs_row_base,
      const uint abs_col_base,
      const uint k_offset_weight_groups,
      const uint k_offset_act_groups,
      const uint weight_groups_per_row,
      const uint act_groups_per_row,
      const thread ThreadContext& thread_context
  ) {
    using Ops = uzu::matmul::MxuFragmentOps<>;
    AccumFragment accumulator;
    accumulator.clear();

    const short2 position = Ops::get_position(thread_context.simd_lane_id);
    thread float* accumulator_elements = accumulator.elements();
    constexpr int k_bytes_per_weight_group = (BITS == 4) ? (int(GROUP_SIZE) / 2) : int(GROUP_SIZE);
    constexpr int act_chunks_per_weight_group = int(GROUP_SIZE) / int(SIMDGROUP_BLOCK_K);

    uzu::matmul::Fragment<int8_t, TILES_N, TILES_K, Ops> ones_tile;
    if constexpr (int8_activation_needs_weight_correction) {
      thread int8_t* ones = ones_tile.elements();
      METAL_PRAGMA_UNROLL
      for (ushort i = 0; i < ones_tile.ELEMENTS_PER_FRAGMENT; ++i) {
        ones[i] = int8_t(1);
      }
    }

    METAL_PRAGMA_NO_UNROLL
    for (int weight_group = 0; weight_group < weight_group_iterations; ++weight_group) {
      const uint weight_group_index = k_offset_weight_groups + uint(weight_group);

      float weight_scale_cache[TILES_N * Ops::THREAD_ELEMENT_COLS];
      fill_column_group_cache<ALIGNED_N, Ops>(
          weight_scale_cache,
          position,
          simdgroup_limit_n,
          abs_col_base,
          weight_groups_per_row,
          weight_group_index,
          [&](uint scale_index) { return static_cast<float>(b_scales[scale_index]); }
      );

      float weight_correction_cache[TILES_N * Ops::THREAD_ELEMENT_COLS];
      if constexpr (int8_activation_needs_weight_correction) {
        fill_column_group_cache<ALIGNED_N, Ops>(
            weight_correction_cache,
            position,
            simdgroup_limit_n,
            abs_col_base,
            weight_groups_per_row,
            weight_group_index,
            [&](uint scale_index) {
              const uint col = scale_index / weight_groups_per_row;
              return int8_weight_correction_coefficient(
                  col,
                  weight_group_index,
                  weight_groups_per_row,
                  b_scales,
                  biases,
                  zero_points
              );
            }
        );
      }

      METAL_PRAGMA_NO_UNROLL
      for (int act_chunk = 0; act_chunk < act_chunks_per_weight_group; ++act_chunk) {
        const int k_element_offset = act_chunk * int(SIMDGROUP_BLOCK_K);
        auto activation_tile = load_int8_left_tile<ALIGNED_M>(
            a_int8_simdgroup + k_element_offset,
            leading_dimension_a,
            simdgroup_limit_m,
            thread_context.simd_lane_id
        );

        uzu::matmul::Fragment<int, TILES_M, TILES_N, Ops> chunk_products;
        chunk_products.clear();
        if constexpr (BITS == 4) {
          uzu::matmul::Fragment<int8_t, TILES_N, TILES_K, Ops> right_tile;
          METAL_PRAGMA_UNROLL
          for (ushort tile_n = 0; tile_n < TILES_N; ++tile_n) {
            METAL_PRAGMA_UNROLL
            for (ushort tile_k = 0; tile_k < TILES_K; ++tile_k) {
              thread auto& weight_vector = right_tile.fragment_at(tile_n, tile_k);
              METAL_PRAGMA_UNROLL
              for (ushort thread_row = 0; thread_row < Ops::THREAD_ELEMENT_ROWS; ++thread_row) {
                const short row = short(tile_n * Ops::FRAGMENT_ROWS) + position.y +
                                  short(thread_row * Ops::THREAD_ELEMENT_ROW_STRIDE);
                const ushort element_base = thread_row * Ops::THREAD_ELEMENT_COLS;
                char4 codes = char4(0);
                if (ALIGNED_N || row < simdgroup_limit_n) {
                  const int k_base = k_element_offset + int(tile_k * Ops::FRAGMENT_COLS) + int(position.x);
                  const ushort packed = *reinterpret_cast<const device ushort*>(
                      b_packed_simdgroup + int(row) * b_row_stride_bytes + (k_base >> 1)
                  );
                  uint spread = uint(packed);
                  spread = (spread | (spread << 8)) & 0x00FF00FFu;
                  spread = (spread | (spread << 4)) & 0x0F0F0F0Fu;
                  codes = as_type<char4>(spread ^ 0x08080808u) - char4(8);
                }
                weight_vector[element_base + 0] = codes.x;
                weight_vector[element_base + 1] = codes.y;
                weight_vector[element_base + 2] = codes.z;
                weight_vector[element_base + 3] = codes.w;
              }
            }
          }
          Ops::template fragment_mma<false, true>(chunk_products, activation_tile, right_tile);
        } else {
          static_assert(BITS == 8, "symmetric int8 activations only support 4-bit or 8-bit weights");
          uzu::matmul::Fragment<int8_t, TILES_N, TILES_K, Ops> right_tile;
          auto right_src = uzu::matmul::fragment_source(
              reinterpret_cast<const device int8_t*>(b_packed_simdgroup) + k_element_offset,
              b_row_stride_bytes
          );
          if constexpr (!ALIGNED_N) {
            right_src = right_src.bounded(simdgroup_limit_n, SIMDGROUP_BLOCK_K);
          }
          right_tile.load_from(thread_context.simd_lane_id, right_src);
          Ops::template fragment_mma<false, true>(chunk_products, activation_tile, right_tile);
        }

        const uint act_group_index = k_offset_act_groups + uint(weight_group * act_chunks_per_weight_group + act_chunk);
        float activation_scale_cache[TILES_M * Ops::THREAD_ELEMENT_ROWS];
        fill_row_group_cache<ALIGNED_M, Ops>(
            activation_scale_cache,
            a_scales,
            position,
            simdgroup_limit_m,
            abs_row_base,
            act_groups_per_row,
            act_group_index
        );

        uzu::matmul::Fragment<int, TILES_M, TILES_N, Ops> activation_row_sums;
        if constexpr (int8_activation_needs_weight_correction) {
          activation_row_sums.clear();
          Ops::template fragment_mma<false, true>(activation_row_sums, activation_tile, ones_tile);
        }
        thread int* chunk_products_data = chunk_products.elements();
        thread int* activation_row_sums_data = activation_row_sums.elements();
        METAL_PRAGMA_UNROLL
        for (ushort tile_row = 0; tile_row < TILES_M; ++tile_row) {
          METAL_PRAGMA_UNROLL
          for (ushort tile_col = 0; tile_col < TILES_N; ++tile_col) {
            const ushort fragment_base = (tile_row * TILES_N + tile_col) * Ops::ELEMENTS_PER_THREAD;
            const short row_base = position.y + tile_row * Ops::FRAGMENT_ROWS;
            const short col_base = position.x + tile_col * Ops::FRAGMENT_COLS;
            METAL_PRAGMA_UNROLL
            for (ushort element = 0; element < Ops::ELEMENTS_PER_THREAD; ++element) {
              const short2 element_offset = Ops::get_element_offset(element);
              const short row = row_base + element_offset.y;
              const short col = col_base + element_offset.x;
              if ((ALIGNED_M || row < simdgroup_limit_m) && (ALIGNED_N || col < simdgroup_limit_n)) {
                const ushort activation_scale_index =
                    tile_row * Ops::THREAD_ELEMENT_ROWS + ushort(element_offset.y / Ops::THREAD_ELEMENT_ROW_STRIDE);
                const ushort weight_scale_index = tile_col * Ops::THREAD_ELEMENT_COLS + ushort(element_offset.x);
                float scaled_products =
                    weight_scale_cache[weight_scale_index] * float(chunk_products_data[fragment_base + element]);
                if constexpr (int8_activation_needs_weight_correction) {
                  scaled_products += weight_correction_cache[weight_scale_index] *
                                     float(activation_row_sums_data[fragment_base + element]);
                }
                accumulator_elements[fragment_base + element] +=
                    activation_scale_cache[activation_scale_index] * scaled_products;
              }
            }
          }
        }
      }

      a_int8_simdgroup += GROUP_SIZE;
      b_packed_simdgroup += k_bytes_per_weight_group;
    }

    return accumulator;
  }

  static METAL_FUNC void run(
      const device AT* a,
      const device BT* b,
      device DT* d,
      const constant uzu::matmul::GemmParams* params,
      GemmAlignment alignment,
      GemmDTransform output_transform,
      const device BT* scales,
      const device BT* biases,
      const device uint8_t* zero_points,
      const device BT* output_bias,
      const device int32_t* rht_factors,
      const device int8_t* a_int8,
      const device float* a_scales,
      threadgroup BT* b_shared,
      const thread ThreadContext& thread_context
  ) {
    const uint partition = thread_context.threadgroup_position.z;
    const uint tile_y = thread_context.threadgroup_position.y;

    const uint2 tile = tile_id(uint2(thread_context.threadgroup_position.x, tile_y), params);
    const auto geometry = ThreadgroupTileGeometry<THREADGROUP_BLOCK_M, THREADGROUP_BLOCK_N>::compute(tile, params);
    if (geometry.out_of_bounds) {
      return;
    }

    const size_t block_row = size_t(geometry.block_row_start);
    const size_t block_col = size_t(geometry.block_col_start);

    const uint k_offset_per_block =
        (B_PROLOGUE == GemmBPrologueKind::FullPrecision) ? uint(THREADGROUP_BLOCK_K_FP) : uint(QUANT_BK);
    const uint k_offset = partition * params->aligned_inner_iterations * k_offset_per_block;

    const device BT* b_block_fp = b + (TRANSPOSE_B ? block_col * params->leading_dimension_b : block_col) +
                                  (TRANSPOSE_B ? k_offset : k_offset * uint(params->leading_dimension_b));

    const ushort tile_row_offset = SIMDGROUP_BLOCK_M * (thread_context.simdgroup_index / SIMDGROUPS_PER_COLUMN);
    const ushort tile_col_offset = SIMDGROUP_BLOCK_N * (thread_context.simdgroup_index % SIMDGROUPS_PER_COLUMN);

    device DT* d_simdgroup = d + size_t(partition) * size_t(params->M) * size_t(params->N) +
                             block_row * params->leading_dimension_d + block_col +
                             tile_row_offset * params->leading_dimension_d + tile_col_offset;

    const short simdgroup_limit_m =
        alignment.contains(GemmAlignment::M)
            ? SIMDGROUP_BLOCK_M
            : short(min(int(SIMDGROUP_BLOCK_M), int(params->M) - int(geometry.block_row_start + tile_row_offset)));
    const short simdgroup_limit_n =
        alignment.contains(GemmAlignment::N)
            ? SIMDGROUP_BLOCK_N
            : short(min(int(SIMDGROUP_BLOCK_N), int(params->N) - int(geometry.block_col_start + tile_col_offset)));

    const device AT* a_simdgroup = a;
    if constexpr (A_PROLOGUE == GemmAPrologueKind::FullPrecision) {
      a_simdgroup +=
          block_row * params->leading_dimension_a + k_offset + size_t(tile_row_offset) * params->leading_dimension_a;
    }
    const device BT* b_simdgroup_fp =
        b_block_fp +
        (TRANSPOSE_B ? size_t(tile_col_offset) * int(params->leading_dimension_b) : size_t(tile_col_offset));

    const ushort tile_block_cols =
        ushort(min(int(THREADGROUP_BLOCK_N), int(params->N) - int(geometry.block_col_start)));

    const bool apply_scale = output_transform.contains(GemmDTransform::SCALE);
    const bool apply_accumulate = output_transform.contains(GemmDTransform::ACCUMULATE);
    const bool apply_bias = output_transform.contains(GemmDTransform::BIAS);

    const device BT* bias_simdgroup = output_bias + size_t(block_col) + size_t(tile_col_offset);

    auto dispatch_aligned_k = [&](auto body) {
      if constexpr (B_PROLOGUE == GemmBPrologueKind::FullPrecision) {
        dispatch_bool(alignment.contains(GemmAlignment::K), body);
      } else {
        body(true_type{});
      }
    };
    dispatch_aligned_k([&](auto aligned_k) {
      dispatch_bool(
          alignment.contains(GemmAlignment::M) || (simdgroup_limit_m == SIMDGROUP_BLOCK_M),
          [&](auto aligned_m) {
            dispatch_bool(
                alignment.contains(GemmAlignment::N) || (simdgroup_limit_n == SIMDGROUP_BLOCK_N),
                [&](auto aligned_n) {
                  auto accumulator_tile = [&]() {
                    if constexpr (A_PROLOGUE == GemmAPrologueKind::Int8Symmetric) {
                      const auto quantized_weights =
                          quantized_weight_addressing(b, block_col, k_offset, int(params->K));
                      const device int8_t* a_int8_simdgroup = a_int8 + block_row * params->leading_dimension_a +
                                                              k_offset +
                                                              size_t(tile_row_offset) * params->leading_dimension_a;
                      const device uint8_t* b_packed_simdgroup =
                          quantized_weights.block + size_t(tile_col_offset) * quantized_weights.row_stride_bytes;
                      return symmetric_int8_activation_k_loop<aligned_m.value, aligned_n.value>(
                          a_int8_simdgroup,
                          b_packed_simdgroup,
                          a_scales,
                          scales,
                          biases,
                          zero_points,
                          int(params->leading_dimension_a),
                          quantized_weights.row_stride_bytes,
                          int(params->aligned_inner_iterations),
                          simdgroup_limit_m,
                          simdgroup_limit_n,
                          uint(geometry.block_row_start) + tile_row_offset,
                          uint(geometry.block_col_start) + tile_col_offset,
                          uint(quantized_weights.k_offset_groups),
                          k_offset / uint(SIMDGROUP_BLOCK_K),
                          uint(quantized_weights.groups_per_row),
                          uint(params->K) / uint(SIMDGROUP_BLOCK_K),
                          thread_context
                      );
                    } else if constexpr (B_PROLOGUE == GemmBPrologueKind::FullPrecision) {
                      const int aligned_k_iterations_fp = int(params->aligned_inner_iterations);
                      return uzu::matmul::gemm_loop<
                          AT,
                          BT,
                          SIMDGROUP_BLOCK_M,
                          SIMDGROUP_BLOCK_N,
                          SIMDGROUP_BLOCK_K,
                          THREADGROUP_BLOCK_K_FP,
                          false,
                          TRANSPOSE_B,
                          aligned_m.value,
                          aligned_n.value,
                          aligned_k.value,
                          AccumulatorType>(
                          a_simdgroup,
                          b_simdgroup_fp,
                          int(params->leading_dimension_a),
                          int(params->leading_dimension_b),
                          int(params->K),
                          aligned_k_iterations_fp,
                          simdgroup_limit_m,
                          simdgroup_limit_n,
                          thread_context
                      );
                    } else {
                      const int aligned_k_iterations_q = int(params->aligned_inner_iterations);
                      const int k_elements = int(params->K);
                      const auto quantized_weights = quantized_weight_addressing(b, block_col, k_offset, k_elements);
                      const int groups_per_row = quantized_weights.groups_per_row;
                      const int k_offset_groups = quantized_weights.k_offset_groups;
                      const device uint8_t* weights_block = quantized_weights.block;
                      const device BT* scales_offset = scales + block_col * groups_per_row + k_offset_groups;

                      auto loader_b = [&]() {
                        if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleBiasDequant) {
                          const device BT* biases_offset = biases + block_col * groups_per_row + k_offset_groups;
                          return BLoaderScaleBias(
                              weights_block,
                              scales_offset,
                              biases_offset,
                              k_elements,
                              b_shared,
                              thread_context.simdgroup_index,
                              thread_context.simd_lane_id
                          );
                        } else if constexpr (B_PROLOGUE == GemmBPrologueKind::ScaleZeroPointDequant) {
                          const int zero_point_stride_per_row =
                              (BITS == 4) ? ((groups_per_row + 1) / 2) : groups_per_row;
                          const device uint8_t* zero_points_row_start =
                              zero_points + block_col * zero_point_stride_per_row +
                              ((BITS == 4) ? (k_offset_groups / 2) : k_offset_groups);
                          return BLoaderScaleZeroPoint(
                              weights_block,
                              scales_offset,
                              zero_points_row_start,
                              k_elements,
                              groups_per_row,
                              b_shared,
                              thread_context.simdgroup_index,
                              thread_context.simd_lane_id
                          );
                        } else {
                          return BLoaderScaleSymmetric(
                              weights_block,
                              scales_offset,
                              nullptr,
                              k_elements,
                              groups_per_row,
                              b_shared,
                              thread_context.simdgroup_index,
                              thread_context.simd_lane_id
                          );
                        }
                      }();

                      return quant_k_loop<aligned_m.value, aligned_n.value>(
                          a_simdgroup,
                          b_shared,
                          int(params->leading_dimension_a),
                          aligned_k_iterations_q,
                          simdgroup_limit_m,
                          simdgroup_limit_n,
                          tile_col_offset,
                          tile_block_cols,
                          loader_b,
                          thread_context
                      );
                    }
                  }();

                  if (apply_scale) {
                    const AccumulatorType scale = AccumulatorType(params->ab_scale);
                    accumulator_tile.map([&](auto value) { return value * scale; });
                  }

                  if (apply_accumulate) {
                    uzu::matmul::Fragment<DT, TILES_M, TILES_N, uzu::matmul::MxuFragmentOps<>> existing_output;
                    auto output_src = uzu::matmul::fragment_source(d_simdgroup, int(params->leading_dimension_d));
                    if constexpr (!(aligned_m.value && aligned_n.value)) {
                      output_src = output_src.bounded(simdgroup_limit_m, simdgroup_limit_n);
                    }
                    existing_output.load_from(thread_context.simd_lane_id, output_src);
                    thread DT* existing_data = existing_output.elements();
                    accumulator_tile.map([&](auto value) { return value + AccumulatorType(*(existing_data++)); });
                  }

                  if (apply_bias) {
                    accumulator_tile.map_coords(thread_context.simd_lane_id, [&](short, short col, auto value) {
                      if constexpr (aligned_n.value) {
                        return value + AccumulatorType(bias_simdgroup[col]);
                      } else {
                        if (col < simdgroup_limit_n) {
                          return value + AccumulatorType(bias_simdgroup[col]);
                        }
                        return value;
                      }
                    });
                  }

                  if constexpr (aligned_m.value && aligned_n.value) {
                    accumulator_tile.store(thread_context.simd_lane_id, d_simdgroup, int(params->leading_dimension_d));
                  } else {
                    accumulator_tile.store_safe(
                        thread_context.simd_lane_id,
                        d_simdgroup,
                        int(params->leading_dimension_d),
                        short2(simdgroup_limit_n, simdgroup_limit_m)
                    );
                  }
                }
            );
          }
      );
    });

    if (output_transform.contains(GemmDTransform::RHT)) {
      threadgroup_barrier(mem_flags::mem_device);
      device DT* d_block = d + block_row * params->leading_dimension_d + block_col;
      const ushort tile_block_rows = ushort(min(int(THREADGROUP_BLOCK_M), int(params->M) - int(block_row)));
      const ushort tile_block_cols = ushort(min(int(THREADGROUP_BLOCK_N), int(params->N) - int(block_col)));
      apply_output_random_hadamard_transform(
          d_block,
          rht_factors + block_col,
          tile_block_rows,
          tile_block_cols,
          params->leading_dimension_d,
          ushort(SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN),
          thread_context
      );
    }
  }
};

} // namespace gemm
} // namespace uzu
