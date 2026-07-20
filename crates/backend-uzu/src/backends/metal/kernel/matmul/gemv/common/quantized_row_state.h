#pragma once

#include "../../../generated/gemm.h"
#include "../../common/defines.h"
#include "../../common/quant_pack.h"

namespace uzu {
namespace gemm {

template <typename U, uint RESULTS_PER_SIMDGROUP>
struct QuantizedAffineRowParams {
  U scale[RESULTS_PER_SIMDGROUP];
  U offset[RESULTS_PER_SIMDGROUP];
};

template <typename T, typename U>
struct QuantizedGroupRows {
  const device T* values;
  uint group_stride;

  QuantizedGroupRows(const device T* values_base, uint base_row, uint group_count, uint group_offset)
      : values(values_base + base_row * group_count + group_offset), group_stride(group_count) {}

  METAL_FUNC U value(uint row) const { return static_cast<U>(values[row * group_stride]); }

  void advance(uint groups) { values += groups; }
};

template <typename BT, typename U, GemmBPrologueKind B_PROLOGUE, uint BITS>
struct QuantizedOffsetState;

template <typename BT, typename U, uint BITS>
struct QuantizedOffsetState<BT, U, GemmBPrologueKind::ScaleBiasDequant, BITS> {
  QuantizedGroupRows<BT, U> biases;

  QuantizedOffsetState(
      const device uint8_t*,
      const device BT* biases_base,
      uint base_row,
      uint group_count,
      uint group_offset
  )
      : biases(biases_base, base_row, group_count, group_offset) {}

  METAL_FUNC U value(uint row, U) const { return biases.value(row); }

  void advance(uint groups) { biases.advance(groups); }
};

template <typename BT, typename U, uint BITS>
struct QuantizedOffsetState<BT, U, GemmBPrologueKind::ScaleZeroPointDequant, BITS> {
  const device uint8_t* zero_points;
  uint zero_point_stride;
  uint zero_point_shift;

  QuantizedOffsetState(
      const device uint8_t* zero_points_base,
      const device BT*,
      uint base_row,
      uint group_count,
      uint group_offset
  ) {
    if constexpr (BITS == 4) {
      zero_point_stride = (group_count + 1) / 2;
      zero_points = zero_points_base + base_row * zero_point_stride + group_offset / 2;
      zero_point_shift = (group_offset & 1) * 4u;
    } else {
      zero_point_stride = group_count;
      zero_points = zero_points_base + base_row * zero_point_stride + group_offset;
    }
  }

  METAL_FUNC U value(uint row, U scale) const {
    uint8_t zero_point_value = zero_points[row * zero_point_stride];
    if constexpr (BITS == 4) {
      zero_point_value = (zero_point_value >> zero_point_shift) & 0x0F;
    }
    return -scale * static_cast<U>(zero_point_value);
  }

  void advance(uint groups) {
    constexpr uint zero_points_per_byte = get_pack_factor<BITS, 8>();
    zero_points += groups / zero_points_per_byte;
  }
};

template <typename BT, typename U, uint BITS>
struct QuantizedOffsetState<BT, U, GemmBPrologueKind::ScaleSymmetricDequant, BITS> {
  QuantizedOffsetState(const device uint8_t*, const device BT*, uint, uint, uint) {}

  METAL_FUNC U value(uint, U scale) const {
    constexpr U midpoint = U(1u << (BITS - 1));
    return -scale * midpoint;
  }
};

template <typename BT, typename U, GemmBPrologueKind B_PROLOGUE, uint BITS, uint RESULTS_PER_SIMDGROUP>
struct QuantizedRowState {
  static_assert(BITS == 4 || BITS == 8, "QMV supports 4- and 8-bit only");

  using Params = QuantizedAffineRowParams<U, RESULTS_PER_SIMDGROUP>;

  QuantizedGroupRows<BT, U> scale_rows;
  QuantizedOffsetState<BT, U, B_PROLOGUE, BITS> offset_state;

  QuantizedRowState(
      const device BT* scales_base,
      const device uint8_t* zero_points_base,
      const device BT* biases_base,
      uint base_row,
      uint group_count,
      uint group_offset
  )
      : scale_rows(scales_base, base_row, group_count, group_offset),
        offset_state(zero_points_base, biases_base, base_row, group_count, group_offset) {}

  // Row index: dense = local row, gather = absolute gather index.
  void load(
      thread Params& params,
      const device uint* gather_indices,
      bool gathered,
      uint batch_idx,
      uint out_vec_size,
      uint out_row
  ) const {
    METAL_PRAGMA_UNROLL
    for (uint row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
      const uint addr_row = gathered ? gather_indices[batch_idx * out_vec_size + out_row + row] : row;
      const U scale = scale_rows.value(addr_row);
      params.scale[row] = scale;
      params.offset[row] = offset_state.value(addr_row, scale);
    }
  }

  void advance(uint groups) {
    scale_rows.advance(groups);
    if constexpr (B_PROLOGUE != GemmBPrologueKind::ScaleSymmetricDequant)
      offset_state.advance(groups);
  }
};

} // namespace gemm
} // namespace uzu
