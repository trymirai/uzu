#pragma once

#include <metal_stdlib>

#include "../../../../common/thread_context.h"
#include "../../../../generated/gemm.h"
#include "../../../common/fragment.h"
#include "../../../common/quant_pack.h"
#include "../../../common/quant_unpack.h"
#include "../schedules/tile_context.h"
#include "cursor.h"

using namespace metal;

namespace uzu {
namespace gemm {
namespace quantized {

enum class Residency {
  Registers,
  Threadgroup,
};

struct Empty {};

template <typename Operand, typename Storage>
static METAL_FUNC float correction_value(
    const thread Storage& storage,
    const float scale,
    const uint scale_index,
    const uint column,
    const uint group,
    const uint groups_per_row
) {
  if constexpr (Operand::SCHEME == GemmBPrologueKind::ScaleBiasDequant) {
    return scale * float(symmetric_zero_point<Operand::BITS>()) + float(storage.bias()[scale_index]);
  } else {
    static_assert(
        Operand::SCHEME == GemmBPrologueKind::ScaleZeroPointDequant,
        "correction is only defined for asymmetric quantized weights"
    );
    if constexpr (Operand::BITS == 8) {
      return scale * (float(symmetric_zero_point<Operand::BITS>()) - float(storage.zp()[scale_index]));
    } else {
      const device uint8_t* row = storage.zp() + column * zero_point_row_stride<Operand::BITS>(groups_per_row);
      return scale *
             (float(symmetric_zero_point<Operand::BITS>()) - float(decode_zero_point<Operand::BITS>(row, group)));
    }
  }
}

template <Residency RESIDENCY, typename Core, typename Storage, typename Operand, bool ALIGNED>
struct Cache;

template <typename Core, typename Storage, typename Operand, bool ALIGNED>
struct Cache<Residency::Registers, Core, Storage, Operand, ALIGNED> {
  using Fragment = typename Core::AccumFragment;
  using Ops = typename Core::FragmentOps;
  using WeightOperand = typename Core::RightOperand;
  UZU_CONST bool ROWS = metal::is_same<Storage, typename Core::LeftStorage>::value;
  UZU_CONST bool NEEDS_CORRECTION = WeightOperand::NEEDS_CORRECTION;

  UZU_CONST ushort TILES = ROWS ? Fragment::ROW_FRAGMENTS : Fragment::COL_FRAGMENTS;
  UZU_CONST ushort SLOTS = ROWS ? Ops::THREAD_ELEMENT_ROWS : Ops::THREAD_ELEMENT_COLS;
  UZU_CONST ushort EXTENT = ROWS ? Fragment::FRAGMENT_ROWS : Fragment::FRAGMENT_COLS;
  UZU_CONST ushort STRIDE = ROWS ? Ops::THREAD_ELEMENT_ROW_STRIDE : 1;

  Storage source;
  short origin;
  short limit;
  uint absolute_base;
  uint scale_k_groups_per_row;
  metal::conditional_t<ROWS, uint, Empty> sum_k_groups_per_row;
  uint first_k_offset;
  float scales[TILES * SLOTS];
  metal::conditional_t<NEEDS_CORRECTION, float[TILES * SLOTS], Empty> corrections;

  METAL_FUNC Cache(
      const Storage source_,
      threadgroup typename Core::RightElementType*,
      const constant uzu::matmul::GemmParams* params,
      const schedules::TileContext tile,
      const thread ThreadContext& thread_context
  ) thread : source(source_) {
    const short2 position = Ops::get_position(thread_context.simd_lane_id);
    if constexpr (ROWS) {
      origin = position.y;
      limit = tile.simdgroup_limit_m;
      absolute_base = tile.abs_row_base;
      scale_k_groups_per_row = uint(params->K) / uint(Operand::GROUP_SIZE);
      const uint correction_group_size =
          Operand::GROUP_SIZE < WeightOperand::GROUP_SIZE ? uint(Operand::GROUP_SIZE) : uint(WeightOperand::GROUP_SIZE);
      sum_k_groups_per_row = uint(params->K) / correction_group_size;
      first_k_offset = tile.k_offset;
    } else {
      origin = position.x;
      limit = tile.simdgroup_limit_n;
      absolute_base = tile.absolute_column_base();
      scale_k_groups_per_row = (uint(params->K) + uint(Operand::GROUP_SIZE) - 1) / uint(Operand::GROUP_SIZE);
      first_k_offset = tile.k_offset;
    }
  }

  METAL_FUNC void fill(const int k_group_index) thread {
    fill_at_k_offset(uint(k_group_index) * uint(WeightOperand::GROUP_SIZE));
  }

  METAL_FUNC void fill_at_k_offset(const uint relative_k_offset) thread {
    const uint absolute_k_offset = first_k_offset + relative_k_offset;
    const uint absolute_k_group = absolute_k_offset / uint(ROWS ? WeightOperand::GROUP_SIZE : Operand::GROUP_SIZE);
    METAL_PRAGMA_UNROLL
    for (ushort tile_index = 0; tile_index < TILES; ++tile_index) {
      METAL_PRAGMA_UNROLL
      for (ushort slot_index = 0; slot_index < SLOTS; ++slot_index) {
        const short coordinate = origin + short(tile_index * EXTENT + slot_index * STRIDE);
        if (ALIGNED || coordinate < limit) {
          const uint line = absolute_base + uint(coordinate);
          uint scale_index;
          if constexpr (ROWS) {
            const uint scale_k_group = absolute_k_offset / uint(Operand::GROUP_SIZE);
            scale_index = line * scale_k_groups_per_row + scale_k_group;
          } else {
            scale_index = line * scale_k_groups_per_row + absolute_k_group;
          }
          const float group_scale = float(source.scales[scale_index]);
          scales[tile_index * SLOTS + slot_index] = group_scale;
          if constexpr (NEEDS_CORRECTION) {
            if constexpr (ROWS) {
              corrections[tile_index * SLOTS + slot_index] =
                  group_scale *
                  float(source.correction_sums()
                            [line * sum_k_groups_per_row +
                             absolute_k_offset / (uint(Operand::GROUP_SIZE) < uint(WeightOperand::GROUP_SIZE)
                                                      ? uint(Operand::GROUP_SIZE)
                                                      : uint(WeightOperand::GROUP_SIZE))]);
            } else {
              corrections[tile_index * SLOTS + slot_index] = correction_value<Operand>(
                  source,
                  group_scale,
                  scale_index,
                  line,
                  absolute_k_group,
                  scale_k_groups_per_row
              );
            }
          }
        }
      }
    }
  }

  METAL_FUNC float scale(const short coordinate) const thread { return scales[slot(coordinate)]; }

  METAL_FUNC float correction(const short coordinate) const thread { return corrections[slot(coordinate)]; }

private:
  METAL_FUNC ushort slot(const short coordinate) const thread {
    const ushort offset = ushort(coordinate - origin);
    return (offset / EXTENT) * SLOTS + (offset % EXTENT) / STRIDE;
  }
};

template <typename Core, typename Storage, typename Operand, bool ALIGNED>
struct Cache<Residency::Threadgroup, Core, Storage, Operand, ALIGNED> {
  using Element = typename Core::RightElementType;
  UZU_CONST bool NEEDS_CORRECTION = Operand::NEEDS_CORRECTION;

  Storage source;
  threadgroup Element* shared;
  const threadgroup Element* staged_scales;
  metal::conditional_t<NEEDS_CORRECTION, const threadgroup float*, Empty> staged_corrections;
  uint k_groups_per_row;
  uint first_k_group;
  uint block_column;
  short staged_offset;
  short block_columns;
  ushort local_thread_index;
  ushort threads_per_threadgroup;

  METAL_FUNC Cache(
      const Storage source_,
      threadgroup Element* shared_,
      const constant uzu::matmul::GemmParams* params,
      const schedules::TileContext tile,
      const thread ThreadContext& thread_context
  ) thread
      : source(source_),
        shared(shared_),
        staged_scales(nullptr),
        k_groups_per_row((uint(params->K) + uint(Operand::GROUP_SIZE) - 1) / uint(Operand::GROUP_SIZE)),
        first_k_group(tile.k_offset / uint(Operand::GROUP_SIZE)),
        block_column(uint(tile.block_col)),
        staged_offset(short(tile.tile_col_offset)),
        block_columns(short(tile.tile_block_cols)),
        local_thread_index(thread_context.simdgroup_index* thread_context.simdgroup_size + thread_context.simd_lane_id),
        threads_per_threadgroup(thread_context.simdgroups_per_threadgroup* thread_context.simdgroup_size) {}

  METAL_FUNC void prefetch(const uint k_group) thread {
    const uint absolute_k_group = first_k_group + k_group;
    uint scale_index = (block_column + uint(local_thread_index)) * k_groups_per_row + absolute_k_group;
    threadgroup Element* slab_scales = shared + slab_offset(k_group);
    if constexpr (NEEDS_CORRECTION) {
      threadgroup float* slab_corrections = correction_slab(k_group);
      for (short column = short(local_thread_index); column < block_columns;
           column += short(threads_per_threadgroup), scale_index += uint(threads_per_threadgroup) * k_groups_per_row) {
        const uint weight_column = block_column + uint(column);
        const float group_scale = float(source.scales[scale_index]);
        slab_scales[column] = Element(group_scale);
        slab_corrections[column] = correction_value<Operand>(
            source,
            group_scale,
            scale_index,
            weight_column,
            absolute_k_group,
            k_groups_per_row
        );
      }
    } else {
      for (short column = short(local_thread_index); column < block_columns;
           column += short(threads_per_threadgroup), scale_index += uint(threads_per_threadgroup) * k_groups_per_row) {
        slab_scales[column] = Element(source.scales[scale_index]);
      }
    }
  }

  METAL_FUNC void fill(const int k_group_index) thread {
    staged_scales = shared + slab_offset(uint(k_group_index));
    if constexpr (NEEDS_CORRECTION) {
      staged_corrections = correction_slab(uint(k_group_index));
    }
  }

  METAL_FUNC float scale(const short column) const thread { return float(staged_scales[staged_offset + column]); }

  METAL_FUNC float correction(const short column) const thread { return staged_corrections[staged_offset + column]; }

private:
  static METAL_FUNC ushort slab_offset(const uint k_group) { return ushort(k_group & 1) * Core::THREADGROUP_BLOCK_N; }

  METAL_FUNC threadgroup float* correction_slab(const uint k_group) const thread {
    return reinterpret_cast<threadgroup float*>(shared + 2 * Core::THREADGROUP_BLOCK_N) + slab_offset(k_group);
  }
};

} // namespace quantized
} // namespace gemm
} // namespace uzu
