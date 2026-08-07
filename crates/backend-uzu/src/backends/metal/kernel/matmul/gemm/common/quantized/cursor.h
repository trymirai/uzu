#pragma once

#include <metal_stdlib>

#include "../../../../common/thread_context.h"
#include "../../../../generated/gemm.h"
#include "../../../common/fragment.h"
#include "../../../common/mxu_fragment/integer_formats.h"
#include "../../../common/quant_pack.h"
#include "../../../common/quant_unpack.h"
#include "../schedules/tile_context.h"

using namespace metal;

namespace uzu {
namespace gemm {
namespace quantized {

using uzu::matmul::DeviceTensorOperand;
using uzu::matmul::IntegerFormat;
using uzu::matmul::Signedness;

using SignedInt4Format = IntegerFormat<4, Signedness::Signed>;
using SignedInt8Format = IntegerFormat<8, Signedness::Signed>;

enum class Axis {
  Rows,
  Columns,
};

template <typename Format, typename Fragment, bool ALIGNED, bool HOIST = true>
struct Cursor;

template <typename Fragment, bool ALIGNED>
METAL_FUNC Fragment
load_int8_tile(const device int8_t* src, const int row_stride, const short simdgroup_limit, const ushort simd_lane_id) {
  Fragment tile;
  auto source = uzu::matmul::fragment_source(src, row_stride);
  if constexpr (!ALIGNED) {
    source = source.bounded(simdgroup_limit, Fragment::COL_FRAGMENTS * Fragment::FragmentOpsType::FRAGMENT_ROWS);
  }
  tile.load_from(simd_lane_id, source);
  return tile;
}

template <typename Fragment>
METAL_FUNC Fragment load_int4_unaligned_tile(
    const device uint8_t* current,
    const int row_stride_bytes,
    const short simdgroup_limit,
    const short2 position
) {
  using Ops = typename Fragment::FragmentOpsType;
  Fragment tile;
  METAL_PRAGMA_UNROLL
  for (ushort tile_n = 0; tile_n < Fragment::ROW_FRAGMENTS; ++tile_n) {
    METAL_PRAGMA_UNROLL
    for (ushort tile_k = 0; tile_k < Fragment::COL_FRAGMENTS; ++tile_k) {
      thread auto& vector = tile.fragment_at(tile_n, tile_k);
      METAL_PRAGMA_UNROLL
      for (ushort thread_row = 0; thread_row < Ops::THREAD_ELEMENT_ROWS; ++thread_row) {
        const short row =
            short(tile_n * Ops::FRAGMENT_ROWS) + position.y + short(thread_row * Ops::THREAD_ELEMENT_ROW_STRIDE);
        const ushort element_base = thread_row * Ops::THREAD_ELEMENT_COLS;
        char4 codes = char4(0);
        if (row < simdgroup_limit) {
          const int k_base = int(tile_k * Ops::FRAGMENT_COLS) + int(position.x);
          const ushort packed =
              *reinterpret_cast<const device ushort*>(current + int(row) * row_stride_bytes + (k_base >> 1));
          codes = unpack_signed_nibbles_to_int8(uint(packed));
        }
        vector[element_base + 0] = codes.x;
        vector[element_base + 1] = codes.y;
        vector[element_base + 2] = codes.z;
        vector[element_base + 3] = codes.w;
      }
    }
  }
  return tile;
}

template <typename Fragment, bool ALIGNED>
struct Cursor<SignedInt8Format, Fragment, ALIGNED, true> {
  using Ops = typename Fragment::FragmentOpsType;
  UZU_CONST short BLOCK_K = short(Fragment::COL_FRAGMENTS * Ops::FRAGMENT_ROWS);

  const device int8_t* current;
  int row_stride;
  short simdgroup_limit;
  ushort simd_lane_id;

  METAL_FUNC Fragment load(const uint) const thread {
    return load_int8_tile<Fragment, ALIGNED>(current, row_stride, simdgroup_limit, simd_lane_id);
  }

  METAL_FUNC void advance() thread { current += BLOCK_K; }

  METAL_FUNC void begin_k_group(const uint) thread {}
};

template <typename Fragment, bool ALIGNED>
struct Cursor<SignedInt8Format, Fragment, ALIGNED, false> {
  using Ops = typename Fragment::FragmentOpsType;
  UZU_CONST short BLOCK_K = short(Fragment::COL_FRAGMENTS * Ops::FRAGMENT_ROWS);

  const device int8_t* base;
  const device int8_t* group_base;
  int row_stride;
  short simdgroup_limit;
  ushort simd_lane_id;

  METAL_FUNC Fragment load(const uint chunk_index) const thread {
    return load_int8_tile<Fragment, ALIGNED>(
        group_base + chunk_index * uint(BLOCK_K),
        row_stride,
        simdgroup_limit,
        simd_lane_id
    );
  }

  METAL_FUNC void advance() thread {}

  METAL_FUNC void begin_k_group(const uint k_offset) thread { group_base = base + k_offset; }
};

template <typename Fragment, bool HOIST>
struct Cursor<SignedInt4Format, Fragment, false, HOIST> {
  using Ops = typename Fragment::FragmentOpsType;
  UZU_CONST short BLOCK_K = short(Fragment::COL_FRAGMENTS * Ops::FRAGMENT_ROWS);

  const device uint8_t* current;
  int row_stride_bytes;
  short simdgroup_limit;
  short2 position;

  METAL_FUNC Fragment load(const uint) const thread {
    return load_int4_unaligned_tile<Fragment>(current, row_stride_bytes, simdgroup_limit, position);
  }

  METAL_FUNC void advance() thread { current += uint(BLOCK_K) * uint(SignedInt4Format::BITS) / 8; }

  METAL_FUNC void begin_k_group(const uint) thread {}
};

template <typename Fragment, bool HOIST>
struct Cursor<SignedInt4Format, Fragment, true, HOIST> {
  using Ops = typename Fragment::FragmentOpsType;
  UZU_CONST short BLOCK_K = short(Fragment::COL_FRAGMENTS * Ops::FRAGMENT_ROWS);

  const device uint8_t* current;
  int row_stride_bytes;

  METAL_FUNC DeviceTensorOperand<SignedInt4Format> load(const uint) const thread { return {current, row_stride_bytes}; }

  METAL_FUNC void advance() thread { current += uint(BLOCK_K) * uint(SignedInt4Format::BITS) / 8; }

  METAL_FUNC void begin_k_group(const uint) thread {}
};

template <Axis AXIS, bool HOIST_OPERAND_ADDRESSING, typename Core, typename Format, bool ALIGNED, typename Storage>
static METAL_FUNC auto make_cursor(
    const Storage source,
    const constant uzu::matmul::GemmParams* params,
    const schedules::TileContext tile,
    const thread ThreadContext& thread_context
) {
  if constexpr (AXIS == Axis::Rows) {
    using Fragment = uzu::matmul::Fragment<int8_t, Core::TILES_M, Core::TILES_K, typename Core::FragmentOps>;
    const device int8_t* current =
        source.codes + size_t(tile.abs_row_base) * params->leading_dimension_a + tile.k_offset;
    if constexpr (HOIST_OPERAND_ADDRESSING) {
      return Cursor<Format, Fragment, ALIGNED, true>{
          current,
          int(params->leading_dimension_a),
          tile.simdgroup_limit_m,
          ushort(thread_context.simd_lane_id)
      };
    } else {
      return Cursor<Format, Fragment, ALIGNED, false>{
          current,
          current,
          int(params->leading_dimension_a),
          tile.simdgroup_limit_m,
          ushort(thread_context.simd_lane_id)
      };
    }
  } else {
    using Ops = uzu::matmul::MxuFragmentOps<>;
    using Fragment = uzu::matmul::Fragment<int8_t, Core::TILES_N, Core::TILES_K, Ops, uzu::matmul::ReadDirect, true>;
    const int row_stride_bytes =
        int(uint(params->K) * uint(get_bytes_per_pack<Format::BITS>()) / uint(get_pack_factor<Format::BITS>()));
    const device uint8_t* current =
        source.codes + size_t(tile.block_col + tile.tile_col_offset) * size_t(row_stride_bytes) +
        size_t(tile.k_offset) * size_t(get_bytes_per_pack<Format::BITS>()) / size_t(get_pack_factor<Format::BITS>());

    if constexpr (Format::BITS == 4 && ALIGNED) {
      return Cursor<Format, Fragment, true, HOIST_OPERAND_ADDRESSING>{current, row_stride_bytes};
    } else if constexpr (Format::BITS == 4) {
      return Cursor<Format, Fragment, false, HOIST_OPERAND_ADDRESSING>{
          current,
          row_stride_bytes,
          tile.simdgroup_limit_n,
          Ops::get_position(thread_context.simd_lane_id)
      };
    } else {
      static_assert(Format::BITS == 8, "integer tile cursors support 4-bit and 8-bit codes");
      const device int8_t* current_int8 = reinterpret_cast<const device int8_t*>(current);
      if constexpr (HOIST_OPERAND_ADDRESSING) {
        return Cursor<Format, Fragment, ALIGNED, true>{
            current_int8,
            row_stride_bytes,
            tile.simdgroup_limit_n,
            ushort(thread_context.simd_lane_id)
        };
      } else {
        return Cursor<Format, Fragment, ALIGNED, false>{
            current_int8,
            current_int8,
            row_stride_bytes,
            tile.simdgroup_limit_n,
            ushort(thread_context.simd_lane_id)
        };
      }
    }
  }
}

} // namespace quantized
} // namespace gemm
} // namespace uzu
