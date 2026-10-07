#pragma once

#include <metal_stdlib>

#include "../../../../common/defines.h"
#include "../../../../generated/gemm.h"
#include "cursor.h"

using namespace metal;

namespace uzu {
namespace gemm {

namespace trellis {

struct GemmTrellisFormat {
  uint32_t vector_width;
  uint32_t transition_bits;
  uint32_t restart_columns;
};

UZU_CONST uint K_STEP = 64, WEIGHT_BIAS = 54;
UZU_CONST uint WEIGHTS_PER_DECODE = 4;  // bytes in a decode_biased_weights uint
UZU_CONST uint MAX_BIASED_WEIGHT = 111; // 8 * 12 + 15
UZU_CONST uint STATE_BITS = 16, BYTE_BITS = 8, V4_VECTOR_WIDTH = 4, V2T6_TRANSITION_BITS = 6;

UZU_CONST uint HASH_MULTIPLIER = 0xCFCCB83Fu, HASH_INCREMENT = 0x584B4AA3u;
UZU_CONST uint HASH_AVALANCHE_MULTIPLIER = 0x85EBCA6Bu, HASH_FOLD_BITS = 16;

UZU_CONST uint TWO_BIT_FIELD_SHIFT = 2, NIBBLE_SHIFT = 4, FIELD_SUM_SHIFT = 3;
UZU_CONST uint TWO_BIT_FIELDS_MASK = 0x33333333u, BYTE_NIBBLES_MASK = 0x0F0F0F0Fu;
UZU_CONST uint FIELD_PAIR_SUM_MULTIPLIER = 3, DITHER_MULTIPLIER = 3;

// Row layout
static METAL_FUNC uint block_bytes(const GemmTrellisFormat format, const uint block_columns) {
  const uint state_count = block_columns / format.vector_width;
  const uint transition_count = state_count - 1;
  return (STATE_BITS + transition_count * format.transition_bits + BYTE_BITS - 1) / BYTE_BITS;
}

static METAL_FUNC uint row_bytes(const GemmTrellisFormat format, const uint code_columns) {
  const uint block_columns = format.restart_columns == 0 ? code_columns : format.restart_columns;
  return code_columns / block_columns * block_bytes(format, block_columns);
}

// MSB-first codes. code_column must be a multiple of 4.
static METAL_FUNC uint2
states_at(const GemmTrellisFormat format, const device uchar* code_row, const uint code_column) {
  if (format.vector_width == V4_VECTOR_WIDTH) {
    const device uchar* block_codes = code_row;
    uint block_column = code_column;
    if (format.restart_columns != 0) {
      block_codes += code_column / format.restart_columns * block_bytes(format, format.restart_columns);
      block_column %= format.restart_columns;
    }
    const device uchar* state_bytes = block_codes + block_column / V4_VECTOR_WIDTH;
    return uint2(uint(state_bytes[0]) << BYTE_BITS | state_bytes[1], 0);
  }
  const uint first_bit = code_column / format.vector_width * format.transition_bits;
  const device uchar* bytes = code_row + first_bit / BYTE_BITS;
  const uint shift = first_bit % BYTE_BITS;
  const uint fourth_byte = format.transition_bits == V2T6_TRANSITION_BITS ? uint(bytes[3]) : 0;
  const uint window = uint(bytes[0]) << 24 | uint(bytes[1]) << 16 | uint(bytes[2]) << 8 | fourth_byte;
  const uint state_mask = (1u << STATE_BITS) - 1;
  const uint first_state = (window >> (16 - shift)) & state_mask;
  const uint second_state = (window >> (16 - format.transition_bits - shift)) & state_mask;
  return uint2(first_state, second_state);
}

// State -> hash -> biased weights
static METAL_FUNC uint hash_without_final_fold(const uint code_state) {
  uint hash = code_state * HASH_MULTIPLIER + HASH_INCREMENT;
  hash ^= hash >> HASH_FOLD_BITS;
  return hash * HASH_AVALANCHE_MULTIPLIER;
}

static METAL_FUNC uint final_fold(const uint hash) { return hash ^ (hash >> HASH_FOLD_BITS); }

// Per byte b: weight + WEIGHT_BIAS = 8 * (sum of b's four 2-bit fields) + ((3 * (b & 15)) & 15).
static METAL_FUNC uint biased_weights(const uint hash_bytes) {
  const uint nibble_field_sums =
      hash_bytes - FIELD_PAIR_SUM_MULTIPLIER * ((hash_bytes >> TWO_BIT_FIELD_SHIFT) & TWO_BIT_FIELDS_MASK);
  const uint byte_field_sums = (nibble_field_sums + (nibble_field_sums >> NIBBLE_SHIFT)) & BYTE_NIBBLES_MASK;
  const uint low_nibble_dither = ((hash_bytes & BYTE_NIBBLES_MASK) * DITHER_MULTIPLIER) & BYTE_NIBBLES_MASK;
  return (byte_field_sums << FIELD_SUM_SHIFT) + low_nibble_dither;
}

static METAL_FUNC uint
decode_biased_weights(const GemmTrellisFormat format, const device uchar* code_row, const uint code_column) {
  const uint2 code_states = states_at(format, code_row, code_column);
  const uint first_hash = hash_without_final_fold(code_states.x);
  if (format.vector_width == V4_VECTOR_WIDTH) {
    return biased_weights(final_fold(first_hash));
  }
  const uint second_hash = hash_without_final_fold(code_states.y);
  const uint low_halves = insert_bits(first_hash, second_hash, HASH_FOLD_BITS, HASH_FOLD_BITS);
  const uint high_halves = insert_bits(second_hash, first_hash >> HASH_FOLD_BITS, 0, HASH_FOLD_BITS);
  return biased_weights(low_halves ^ high_halves);
}

} // namespace trellis

namespace quantized {

template <typename Fragment, bool ALIGNED>
struct TrellisCursor {
  using Ops = typename Fragment::FragmentOpsType;
  UZU_CONST short BLOCK_K = short(Fragment::COL_FRAGMENTS * Ops::FRAGMENT_ROWS);

  const device uint8_t* code_rows;
  uint code_row_stride_bytes;
  short tile_row_limit;
  short2 position;
  uint k_offset;
  trellis::GemmTrellisFormat format;

  METAL_FUNC Fragment load(const uint) const thread {
    const short row_limit = tile_row_limit - position.y;
    Fragment tile;
    for_each_fragment_row<Fragment>([&](ushort fragment_row, ushort row_slot, short row_offset) {
      const bool live = ALIGNED || row_offset < row_limit;
      const device uint8_t* code_row = code_rows + int(row_offset) * int(code_row_stride_bytes);
      METAL_PRAGMA_UNROLL
      for (ushort k_fragment = 0; k_fragment < Fragment::COL_FRAGMENTS; ++k_fragment) {
        reinterpret_cast<thread uint*>(&tile.fragment_at(fragment_row, k_fragment))[row_slot] =
            live ? trellis::decode_biased_weights(
                       format,
                       code_row,
                       k_offset + uint(k_fragment * Ops::FRAGMENT_ROWS) + uint(position.x)
                   )
                 : 0u;
      }
    });
    return tile;
  }

  METAL_FUNC void advance() thread { k_offset += uint(BLOCK_K); }
};

} // namespace quantized
} // namespace gemm
} // namespace uzu
