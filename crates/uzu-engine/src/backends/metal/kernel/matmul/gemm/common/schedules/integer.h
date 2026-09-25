#pragma once

#include <metal_stdlib>

#include "../../../../common/integral_constant.h"
#include "../../../../common/thread_context.h"
#include "../../../../generated/gemm.h"
#include "../../../../generated/matmul.h"
#include "../../../common/fragment.h"
#include "../../../common/mxu_fragment/ops.h"
#include "../gemm_alignment.h"
#include "../gemm_tiling.h"
#include "../operands.h"
#include "../quantized/cursor.h"
#include "tile_context.h"

using namespace metal;

namespace uzu {
namespace gemm {

namespace schedules {

namespace {

static METAL_FUNC uint clamp_group_output_column(const uint column, const uint padded_output_count) {
  return min(column, padded_output_count - uzu::matmul::QUANT_PARAMS_GROUP_OUTPUT_ALIGNMENT);
}

template <int COUNT, typename Visitor>
static METAL_FUNC void for_each_static_index(Visitor visitor) {
  const_for_loop<0, COUNT, 1>([&](auto slot) { visitor(ushort(decltype(slot)::value)); });
}

template <typename Accumulator, typename GroupProductFragment, typename Visitor>
static METAL_FUNC void for_each_fragment(
    thread Accumulator& accumulator,
    thread GroupProductFragment& group_product,
    Visitor visitor
) {
  for_each_static_index<int(GroupProductFragment::COL_FRAGMENTS)>([&](const ushort tile_n) {
    for_each_static_index<int(GroupProductFragment::ROW_FRAGMENTS)>([&](const ushort tile_m) {
      visitor(tile_m, tile_n, accumulator.fragment_at(tile_m, tile_n), group_product.fragment_at(tile_m, tile_n));
    });
  });
}

} // namespace

struct MetadataContext {
  uint left_row_base;
  short left_row_limit;
  uint right_column_base;
  short right_column_limit;
  uint left_group_count;
  uint right_group_count;
  uint right_scale_group_stride;
  uint right_zero_point_group_stride;
};

template <typename LeftOperand, typename RightOperand>
struct IntegerSchedule {
  UZU_CONST uint RIGHT_GROUP_SIZE = uint(RightOperand::GROUP_SIZE);
  UZU_CONST bool HAS_ZERO_POINTS = RightOperand::SCHEME == GemmBPrologueKind::ScaleZeroPointDequant;
  UZU_CONST bool HAS_BIAS = RightOperand::SCHEME == GemmBPrologueKind::ScaleBiasDequant;
  UZU_CONST uchar RIGHT_CODE_OFFSET = uchar(RightOperand::CODE_ORIGIN);

  template <typename Core, bool ALIGNED_M>
  struct LeftMetadata {
    using Ops = typename Core::FragmentOps;
    UZU_CONST ushort THREAD_ROWS_PER_FRAGMENT = Ops::THREAD_ELEMENT_ROWS;
    UZU_CONST ushort LEFT_VALUE_COUNT = Core::TILES_M * THREAD_ROWS_PER_FRAGMENT;

    float scales[LEFT_VALUE_COUNT];
    metal::conditional_t<HAS_ZERO_POINTS, int, float> group_corrections[LEFT_VALUE_COUNT];

    struct LeftSlot {
      uint row_index;
      bool live;
    };

    static METAL_FUNC LeftSlot left_slot(const MetadataContext metadata_context, const ushort index) {
      const short left_row_offset = short(
          (index / THREAD_ROWS_PER_FRAGMENT) * Ops::FRAGMENT_ROWS +
          (index % THREAD_ROWS_PER_FRAGMENT) * Ops::THREAD_ELEMENT_ROW_STRIDE
      );
      return {
          metadata_context.left_row_base + uint(left_row_offset),
          ALIGNED_M || (left_row_offset < metadata_context.left_row_limit)
      };
    }

    METAL_FUNC void load(
        const typename Core::LeftStorage left,
        const MetadataContext metadata_context,
        const uint k_offset
    ) thread {
      if (LeftOperand::GROUP_SIZE == RightOperand::GROUP_SIZE || (k_offset % uint(LeftOperand::GROUP_SIZE)) == 0u) {
        const uint left_group_index = k_offset / uint(LeftOperand::GROUP_SIZE);
        for_each_static_index<int(LEFT_VALUE_COUNT)>([&](const ushort index) {
          const LeftSlot slot = left_slot(metadata_context, index);
          scales[index] =
              slot.live ? float(left.scales[slot.row_index * metadata_context.left_group_count + left_group_index])
                        : 0.0f;
        });
      }

      if constexpr (HAS_ZERO_POINTS || HAS_BIAS) {
        const uint right_group_index = k_offset / RIGHT_GROUP_SIZE;
        for_each_static_index<int(LEFT_VALUE_COUNT)>([&](const ushort index) {
          const LeftSlot slot = left_slot(metadata_context, index);
          const int code_sum =
              slot.live
                  ? left.correction_sums()[slot.row_index * metadata_context.right_group_count + right_group_index]
                  : 0;
          if constexpr (HAS_ZERO_POINTS) {
            group_corrections[index] = code_sum;
          } else {
            group_corrections[index] = slot.live ? (scales[index] * float(code_sum)) : 0.0f;
          }
        });
      }
    }
  };

  template <typename Core, bool ALIGNED_N>
  struct RightMetadata {
    using Ops = typename Core::FragmentOps;
    using ScaleElement = typename RightOperand::ScaleElement;
    using ScaleVector = vec<ScaleElement, Ops::THREAD_ELEMENT_COLS>;
    using ZeroPointVector = uchar4;
    static_assert(
        Ops::THREAD_ELEMENT_COLS == uzu::matmul::QUANT_PARAMS_GROUP_OUTPUT_ALIGNMENT,
        "group-major metadata must use the metadata load width"
    );

    ScaleVector scales[Core::TILES_N];
    ScaleVector bias_offsets[Core::TILES_N];
    ZeroPointVector zero_points[Core::TILES_N];

    METAL_FUNC void load(
        const typename Core::RightStorage right,
        const MetadataContext metadata_context,
        const uint right_group_index
    ) thread {
      for_each_static_index<int(Core::TILES_N)>([&](const ushort tile_n) {
        const ushort right_column_offset = tile_n * Ops::FRAGMENT_COLS;
        uint right_scale_column_start = metadata_context.right_column_base + uint(right_column_offset);
        if constexpr (!ALIGNED_N) {
          right_scale_column_start =
              clamp_group_output_column(right_scale_column_start, metadata_context.right_scale_group_stride);
        }
        scales[tile_n] = *reinterpret_cast<const device ScaleVector*>(
            right.scales + right_group_index * metadata_context.right_scale_group_stride + right_scale_column_start
        );
        if constexpr (HAS_BIAS) {
          bias_offsets[tile_n] = *reinterpret_cast<const device ScaleVector*>(
              right.bias() + right_group_index * metadata_context.right_scale_group_stride + right_scale_column_start
          );
          bias_offsets[tile_n] += scales[tile_n] * ScaleElement(RIGHT_CODE_OFFSET);
        }
        if constexpr (HAS_ZERO_POINTS) {
          constexpr uint ZERO_POINT_PACK_FACTOR = 8u / uint(RightOperand::BITS);
          const device uint8_t* zero_point_row =
              right.zp() +
              right_group_index * (metadata_context.right_zero_point_group_stride / ZERO_POINT_PACK_FACTOR);
          uint right_zero_point_column_start = metadata_context.right_column_base + uint(right_column_offset);
          if constexpr (!ALIGNED_N) {
            right_zero_point_column_start = clamp_group_output_column(
                right_zero_point_column_start,
                metadata_context.right_zero_point_group_stride
            );
          }
          if constexpr (RightOperand::BITS == 4) {
            const ushort packed =
                *reinterpret_cast<const device ushort*>(zero_point_row + (right_zero_point_column_start >> 1));
            uint spread = (uint(packed) | (uint(packed) << 8)) & 0x00FF00FFu;
            spread = (spread | (spread << 4)) & 0x0F0F0F0Fu;
            zero_points[tile_n] = as_type<ZeroPointVector>(spread);
          } else {
            zero_points[tile_n] =
                *reinterpret_cast<const device ZeroPointVector*>(zero_point_row + right_zero_point_column_start);
          }
        }
        if constexpr (!ALIGNED_N) {
          const bool4 live =
              (short4(right_column_offset) + short4(0, 1, 2, 3)) < short4(metadata_context.right_column_limit);
          scales[tile_n] = select(ScaleVector(0), scales[tile_n], live);
          if constexpr (HAS_BIAS) {
            bias_offsets[tile_n] = select(ScaleVector(0), bias_offsets[tile_n], live);
          }
          if constexpr (HAS_ZERO_POINTS) {
            zero_points[tile_n] = select(ZeroPointVector(RIGHT_CODE_OFFSET), zero_points[tile_n], live);
          }
        }
      });
    }
  };

  template <typename Core>
  using GroupProducts = uzu::matmul::Fragment<int, Core::TILES_M, Core::TILES_N, typename Core::FragmentOps>;

  template <typename Core, typename LeftCodes, typename RightCodes>
  static METAL_FUNC GroupProducts<Core> multiply_group(thread LeftCodes& left_codes, thread RightCodes& right_codes) {
    constexpr int chunks_per_group = int(RIGHT_GROUP_SIZE) / int(Core::SIMDGROUP_BLOCK_K);
    constexpr bool PREFETCH_INT4_CHUNKS = Core::TILES_M == 1 && chunks_per_group > 1 && RightOperand::BITS == 4;
    GroupProducts<Core> group_product;
    group_product.clear();
    if constexpr (PREFETCH_INT4_CHUNKS) {
      typename RightCodes::PackedChunk packed_chunks[chunks_per_group];
      for_each_static_index<chunks_per_group>([&](const ushort chunk) {
        packed_chunks[chunk] = right_codes.fetch(uint(chunk));
      });
      for_each_static_index<chunks_per_group>([&](const ushort chunk) {
        auto left_tile = left_codes.load(uint(chunk));
        auto right_tile = right_codes.decode(packed_chunks[chunk]);
        uzu::matmul::fragment_mma(group_product, left_tile, right_tile);
        left_codes.advance();
        right_codes.advance();
      });
    } else {
      METAL_PRAGMA_NO_UNROLL
      for (int chunk = 0; chunk < chunks_per_group; ++chunk) {
        auto left_tile = left_codes.load(uint(chunk));
        auto right_tile = right_codes.load(uint(chunk));
        uzu::matmul::fragment_mma(group_product, left_tile, right_tile);
        left_codes.advance();
        right_codes.advance();
      }
    }
    return group_product;
  }

  template <typename Core, bool ALIGNED_M, bool ALIGNED_N>
  static METAL_FUNC void accumulate_group(
      thread typename Core::AccumFragment& accumulator,
      thread GroupProducts<Core>& group_product,
      const thread LeftMetadata<Core, ALIGNED_M>& left_metadata,
      const thread RightMetadata<Core, ALIGNED_N>& right_metadata
  ) {
    using Ops = typename Core::FragmentOps;

    for_each_fragment(
        accumulator,
        group_product,
        [&](const ushort tile_m, const ushort tile_n, thread auto& accumulated, thread auto& group_product_value) {
          for_each_static_index<int(Ops::THREAD_ELEMENT_ROWS) * int(Ops::THREAD_ELEMENT_COLS)>(
              [&](const ushort element) {
                const ushort right_column = element % Ops::THREAD_ELEMENT_COLS;
                const ushort left_row = tile_m * Ops::THREAD_ELEMENT_ROWS + element / Ops::THREAD_ELEMENT_COLS;
                const float right_scale = float(right_metadata.scales[tile_n][right_column]);
                int centered_product = group_product_value[element];
                if constexpr (HAS_ZERO_POINTS) {
                  centered_product += (int(RIGHT_CODE_OFFSET) - int(right_metadata.zero_points[tile_n][right_column])) *
                                      left_metadata.group_corrections[left_row];
                }
                accumulated[element] =
                    fma(left_metadata.scales[left_row] * right_scale, float(centered_product), accumulated[element]);
                if constexpr (HAS_BIAS) {
                  accumulated[element] =
                      fma(right_metadata.bias_offsets[tile_n][right_column],
                          left_metadata.group_corrections[left_row],
                          accumulated[element]);
                }
              }
          );
        }
    );
  }

  template <typename Core, bool ALIGNED_M, bool ALIGNED_N>
  static METAL_FUNC typename Core::AccumFragment launch(
      typename Core::LeftStorage left_storage,
      typename Core::RightStorage right_storage,
      threadgroup typename Core::RightElementType*,
      const constant uzu::matmul::GemmParams* params,
      const TileContext tile,
      const GemmAlignment,
      const thread ThreadContext& thread_context
  ) {
    static_assert(LeftOperand::QUANTIZED && RightOperand::QUANTIZED, "integer schedule requires quantized operands");
    static_assert(RightOperand::GROUP_SIZE % Core::SIMDGROUP_BLOCK_K == 0, "right groups must contain MMA chunks");
    static_assert(
        LeftOperand::GROUP_SIZE % RightOperand::GROUP_SIZE == 0,
        "the left group must hold a whole number of right groups"
    );

    constexpr bool HOIST_OPERAND_ADDRESSING =
        !(RightOperand::SCHEME == GemmBPrologueKind::ScaleSymmetricDequant &&
          Core::TILING == GemmTiling::Tile128x128x256_Simdgroups4x4);

    auto left_codes = quantized::make_left_cursor<HOIST_OPERAND_ADDRESSING, Core, LeftOperand, ALIGNED_M>(
        left_storage,
        params,
        tile,
        thread_context
    );
    auto right_codes = quantized::make_right_cursor<HOIST_OPERAND_ADDRESSING, Core, RightOperand, ALIGNED_N>(
        right_storage,
        params,
        tile,
        thread_context
    );

    const short2 position = Core::FragmentOps::get_position(thread_context.simd_lane_id);
    const MetadataContext metadata_context = {
        tile.abs_row_base + uint(position.y),
        short(tile.simdgroup_limit_m - position.y),
        tile.absolute_column_base() + uint(position.x),
        short(tile.simdgroup_limit_n - position.x),
        uint(params->K) / uint(LeftOperand::GROUP_SIZE),
        uint(params->K) / RIGHT_GROUP_SIZE,
        params->scale_group_stride,
        params->zero_point_group_stride,
    };
    const uint first_right_group = tile.k_offset / RIGHT_GROUP_SIZE;

    LeftMetadata<Core, ALIGNED_M> left_metadata;
    RightMetadata<Core, ALIGNED_N> right_metadata;
    typename Core::AccumFragment accumulator;
    accumulator.clear();

    const int right_group_count = int(params->aligned_inner_iterations);
    METAL_PRAGMA_NO_UNROLL
    for (int right_group_index = 0; right_group_index < right_group_count; ++right_group_index) {
      const uint absolute_right_group = first_right_group + uint(right_group_index);
      const uint right_group_offset = uint(right_group_index * int(RIGHT_GROUP_SIZE));
      left_codes.begin_k_group(right_group_offset);
      right_codes.begin_k_group(right_group_offset);
      auto group_product = multiply_group<Core>(left_codes, right_codes);
      left_metadata.load(left_storage, metadata_context, tile.k_offset + right_group_offset);
      right_metadata.load(right_storage, metadata_context, absolute_right_group);
      accumulate_group<Core, ALIGNED_M, ALIGNED_N>(accumulator, group_product, left_metadata, right_metadata);
    }

    return accumulator;
  }
};

} // namespace schedules
} // namespace gemm
} // namespace uzu
