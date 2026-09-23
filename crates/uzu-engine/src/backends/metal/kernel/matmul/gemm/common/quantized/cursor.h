#pragma once

#include <metal_stdlib>

#include "../../../../common/thread_context.h"
#include "../../../../generated/gemm.h"
#include "../../../common/fragment.h"
#include "../../../common/quant_pack.h"
#include "../../../common/quant_unpack.h"
#include "../schedules/tile_context.h"

using namespace metal;

namespace uzu {
namespace gemm {
namespace quantized {

namespace {

template <typename Fragment, typename Visitor>
static METAL_FUNC void for_each_fragment_row(Visitor visitor) {
  using Ops = typename Fragment::FragmentOpsType;
  METAL_PRAGMA_UNROLL
  for (ushort fragment_row = 0; fragment_row < Fragment::ROW_FRAGMENTS; ++fragment_row) {
    METAL_PRAGMA_UNROLL
    for (ushort row_slot = 0; row_slot < Ops::THREAD_ELEMENT_ROWS; ++row_slot) {
      const short row_offset =
          short(fragment_row * Ops::FRAGMENT_ROWS) + short(row_slot * Ops::THREAD_ELEMENT_ROW_STRIDE);
      visitor(fragment_row, row_slot, row_offset);
    }
  }
}
} // namespace

template <typename Fragment, bool ALIGNED, bool CODES_GROUPED_BY_NIBBLE>
METAL_FUNC Fragment
load_int8_tile(const device int8_t* src, const int row_stride, const short simdgroup_limit, const ushort simd_lane_id) {
  Fragment tile;
  if constexpr (CODES_GROUPED_BY_NIBBLE) {
    using Ops = typename Fragment::FragmentOpsType;
    const short2 position = Ops::get_position(simd_lane_id);
    const device int8_t* base = src + int(position.y) * row_stride + int(get_pack_factor<W4_BITS>()) * int(position.x);
    const short row_limit = simdgroup_limit - position.y;
    for_each_fragment_row<Fragment>([&](ushort fragment_row, ushort row_slot, short row_offset) {
      vec<uint, Fragment::COL_FRAGMENTS> packed_chunk(0u);
      if (ALIGNED || row_offset < row_limit) {
        packed_chunk =
            *reinterpret_cast<const device vec<uint, Fragment::COL_FRAGMENTS>*>(base + int(row_offset) * row_stride);
      }
      METAL_PRAGMA_UNROLL
      for (ushort fragment_column = 0; fragment_column < Fragment::COL_FRAGMENTS; ++fragment_column) {
        reinterpret_cast<thread uint*>(&tile.fragment_at(fragment_row, fragment_column))[row_slot] =
            packed_chunk[fragment_column];
      }
    });
  } else {
    auto source = uzu::matmul::fragment_source(src, row_stride);
    if constexpr (!ALIGNED) {
      source = source.bounded(simdgroup_limit, Fragment::COL_FRAGMENTS * Fragment::FragmentOpsType::FRAGMENT_ROWS);
    }
    tile.load_from(simd_lane_id, source);
  }
  return tile;
}

template <typename Fragment, bool ALIGNED, bool CODES_GROUPED_BY_NIBBLE, bool HOISTED>
struct Int8Cursor {
  using Ops = typename Fragment::FragmentOpsType;
  UZU_CONST short BLOCK_K = short(Fragment::COL_FRAGMENTS * Ops::FRAGMENT_ROWS);

  const device int8_t* origin;
  const device int8_t* address;
  int row_stride;
  short simdgroup_limit;
  ushort simd_lane_id;

  METAL_FUNC Fragment load(const uint chunk_index) const thread {
    const device int8_t* source = address;
    if constexpr (!HOISTED) {
      source += chunk_index * uint(BLOCK_K);
    }
    return load_int8_tile<Fragment, ALIGNED, CODES_GROUPED_BY_NIBBLE>(
        source,
        row_stride,
        simdgroup_limit,
        simd_lane_id
    );
  }

  METAL_FUNC void advance() thread {
    if constexpr (HOISTED) {
      address += BLOCK_K;
    }
  }

  METAL_FUNC void begin_k_group(const uint k_offset) thread {
    if constexpr (!HOISTED) {
      address = origin + k_offset;
    }
  }
};

template <typename Fragment, bool ALIGNED, ushort CODE_ORIGIN>
struct W4Cursor {
  using Ops = typename Fragment::FragmentOpsType;
  UZU_CONST short BLOCK_K = short(Fragment::COL_FRAGMENTS * Ops::FRAGMENT_ROWS);

  UZU_CONST uint CHUNK_BYTES = uint(BLOCK_K) * uint(get_bytes_per_pack<W4_BITS>()) / uint(get_pack_factor<W4_BITS>());

  struct PackedChunk {
    UZU_CONST ushort WORD_COUNT = Fragment::ROW_FRAGMENTS * Ops::THREAD_ELEMENT_ROWS;
    uint words[WORD_COUNT];

    static constexpr ushort word_index(const ushort fragment_row, const ushort row_slot) {
      return fragment_row * Ops::THREAD_ELEMENT_ROWS + row_slot;
    }
  };

  const device uint8_t* current;
  int row_stride_bytes;
  short tile_row_limit;
  short2 position;
  bool signed_codes;

  METAL_FUNC PackedChunk fetch(const uint chunk_index) const thread {
    const uint padding_word = (CODE_ORIGIN == 0) == signed_codes ? W4_SIGN_MASK : 0u;
    const device uint8_t* base =
        current + chunk_index * CHUNK_BYTES + int(position.y) * row_stride_bytes + int(position.x);
    const short row_limit = tile_row_limit - position.y;
    PackedChunk packed_chunk;
    for_each_fragment_row<Fragment>([&](ushort fragment_row, ushort row_slot, short row_offset) {
      uint word = padding_word;
      if (ALIGNED || row_offset < row_limit) {
        word = *reinterpret_cast<const device uint*>(base + int(row_offset) * row_stride_bytes);
      }
      packed_chunk.words[PackedChunk::word_index(fragment_row, row_slot)] = word;
    });
    return packed_chunk;
  }

  METAL_FUNC Fragment decode(const thread PackedChunk& packed_chunk) const thread {
    Fragment tile;
    for_each_fragment_row<Fragment>([&](ushort fragment_row, ushort row_slot, short) {
      const uint word =
          packed_chunk.words[PackedChunk::word_index(fragment_row, row_slot)] ^ (signed_codes ? W4_SIGN_MASK : 0u);
      uint low = word & W4_NIBBLE_MASK;
      uint high = (word >> W4_BITS) & W4_NIBBLE_MASK;
      if constexpr (CODE_ORIGIN != 0) {
        low = as_type<uint>(as_type<char4>(low) - char4(char(CODE_ORIGIN)));
        high = as_type<uint>(as_type<char4>(high) - char4(char(CODE_ORIGIN)));
      }
      reinterpret_cast<thread uint*>(&tile.fragment_at(fragment_row, 0))[row_slot] = low;
      reinterpret_cast<thread uint*>(&tile.fragment_at(fragment_row, 1))[row_slot] = high;
    });
    return tile;
  }

  METAL_FUNC Fragment load(const uint) const thread { return decode(fetch(0)); }

  METAL_FUNC void advance() thread { current += CHUNK_BYTES; }

  METAL_FUNC void begin_k_group(const uint) thread {}
};

template <bool HOIST_OPERAND_ADDRESSING, typename Core, typename LeftOperand, bool ALIGNED>
static METAL_FUNC auto make_left_cursor(
    const typename Core::LeftStorage source,
    const constant uzu::matmul::GemmParams* params,
    const schedules::TileContext tile,
    const thread ThreadContext& thread_context
) {
  using Fragment = uzu::matmul::Fragment<int8_t, Core::TILES_M, Core::TILES_K, typename Core::FragmentOps>;
  const device int8_t* origin = source.codes + size_t(tile.abs_row_base) * params->leading_dimension_a + tile.k_offset;
  return Int8Cursor<Fragment, ALIGNED, LeftOperand::GROUPED_BY_NIBBLE, HOIST_OPERAND_ADDRESSING>{
      origin,
      origin,
      int(params->leading_dimension_a),
      tile.simdgroup_limit_m,
      ushort(thread_context.simd_lane_id)
  };
}

template <bool HOIST_OPERAND_ADDRESSING, typename Core, typename Operand, bool ALIGNED>
static METAL_FUNC auto make_right_cursor(
    const typename Core::RightStorage source,
    const constant uzu::matmul::GemmParams* params,
    const schedules::TileContext tile,
    const thread ThreadContext& thread_context
) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using Fragment = uzu::matmul::Fragment<int8_t, Core::TILES_N, Core::TILES_K, Ops, uzu::matmul::ReadDirect, true>;
  const int row_stride_bytes =
      int(uint(params->K) * uint(get_bytes_per_pack<Operand::BITS>()) / uint(get_pack_factor<Operand::BITS>()));
  const device uint8_t* current =
      source.codes + size_t(tile.block_col + tile.tile_col_offset) * size_t(row_stride_bytes) +
      size_t(tile.k_offset) * size_t(get_bytes_per_pack<Operand::BITS>()) / size_t(get_pack_factor<Operand::BITS>());

  if constexpr (Operand::BITS == 4) {
    static_assert(Core::TILES_K == get_pack_factor<Operand::BITS>(), "W4 requires two K fragments");
    return W4Cursor<Fragment, ALIGNED, Operand::CODE_ORIGIN>{
        current,
        row_stride_bytes,
        tile.simdgroup_limit_n,
        Ops::get_position(thread_context.simd_lane_id),
        source.signed_codes
    };
  } else {
    static_assert(Operand::BITS == 8, "integer tile cursors support 4-bit and 8-bit codes");
    const device int8_t* origin_int8 = reinterpret_cast<const device int8_t*>(current);
    return Int8Cursor<Fragment, ALIGNED, false, HOIST_OPERAND_ADDRESSING>{
        origin_int8,
        origin_int8,
        row_stride_bytes,
        tile.simdgroup_limit_n,
        ushort(thread_context.simd_lane_id)
    };
  }
}

} // namespace quantized
} // namespace gemm
} // namespace uzu
