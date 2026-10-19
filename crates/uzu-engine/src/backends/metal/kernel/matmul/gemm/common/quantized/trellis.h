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
UZU_CONST uint STATE_BITS = 16, BYTE_BITS = 8, V4_VECTOR_WIDTH = 4, V2T6_TRANSITION_BITS = 6, ROW_ALIGNMENT = 16;

UZU_CONST uint HASH_MULTIPLIER = 0xCFCCB83Fu, HASH_INCREMENT = 0x584B4AA3u;
UZU_CONST uint HASH_AVALANCHE_MULTIPLIER = 0x85EBCA6Bu, HASH_FOLD_BITS = 16;

UZU_CONST uint TWO_BIT_FIELD_SHIFT = 2, NIBBLE_SHIFT = 4, FIELD_SUM_SHIFT = 3;
UZU_CONST uint TWO_BIT_FIELDS_MASK = 0x33333333u, BYTE_NIBBLES_MASK = 0x0F0F0F0Fu;
UZU_CONST uint FIELD_PAIR_SUM_MULTIPLIER = 3, DITHER_MULTIPLIER = 3;

// Row layout: restart blocks sit back to back and the row pads to 16 bytes (an unaligned row start costs up to 16%
// at M <= 8); a connected tape pads to a byte.
static METAL_FUNC uint block_bits(const GemmTrellisFormat format, const uint block_columns) {
  const uint state_count = block_columns / format.vector_width;
  const uint transition_count = state_count - 1;
  return STATE_BITS + transition_count * format.transition_bits;
}

static METAL_FUNC uint row_bytes(const GemmTrellisFormat format, const uint code_columns) {
  if (format.restart_columns == 0) {
    return (block_bits(format, code_columns) + BYTE_BITS - 1) / BYTE_BITS;
  }
  const uint stream_bits = code_columns / format.restart_columns * block_bits(format, format.restart_columns);
  const uint stream_bytes = (stream_bits + BYTE_BITS - 1) / BYTE_BITS;
  return (stream_bytes + ROW_ALIGNMENT - 1) / ROW_ALIGNMENT * ROW_ALIGNMENT;
}

// MSB-first codes. code_column must be a multiple of 4.
static METAL_FUNC uint2
states_at(const GemmTrellisFormat format, const device uchar* code_row, const uint code_column) {
  if (format.vector_width == V4_VECTOR_WIDTH) {
    uint block_first_bit = 0;
    uint block_column = code_column;
    if (format.restart_columns != 0) {
      block_first_bit = code_column / format.restart_columns * block_bits(format, format.restart_columns);
      block_column %= format.restart_columns;
    }
    const uint first_bit = block_first_bit + block_column / V4_VECTOR_WIDTH * format.transition_bits;
    const uint byte = first_bit / BYTE_BITS;
    const uint shift = first_bit % BYTE_BITS;
    // Restart rows hold little-endian words, so stream byte i sits at i ^ 3.
    const uint swizzle = format.restart_columns != 0 ? 3 : 0;
    // A state at a byte boundary may be the row's last; any other one spans three bytes of its row.
    const uint third_byte = shift == 0 ? 0 : uint(code_row[(byte + 2) ^ swizzle]);
    const uint window = uint(code_row[byte ^ swizzle]) << 16 | uint(code_row[(byte + 1) ^ swizzle]) << 8 | third_byte;
    return uint2((window >> (BYTE_BITS - shift)) & ((1u << STATE_BITS) - 1), 0);
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

static METAL_FUNC uint state_weights(const uint state) {
  return biased_weights(final_fold(hash_without_final_fold(state)));
}

// Columns code_column..code_column + 15 (code_column a multiple of 16) of a V4 restart tape: four states of one block
// from one 64-bit window of little-endian words. Needs 6 <= transition_bits <= 8 and 16-byte rows. T6 and T8 windows
// end by bit 64; T7 windows can pass it and always load a third word, from a selected index (an unused load re-reads
// the middle word), which is cheaper than a conditional load.
static METAL_FUNC uint4 v4_decode_biased_weight_quad(
    const GemmTrellisFormat format,
    const device uchar* codes,
    const uint row_byte,
    const uint code_column
) {
  const uint transition_bits = format.transition_bits;
  const uint first_bit = code_column / format.restart_columns * block_bits(format, format.restart_columns) +
                         code_column % format.restart_columns / V4_VECTOR_WIDTH * transition_bits;
  const uint byte = row_byte + first_bit / BYTE_BITS;
  const uint shift = byte % 4 * BYTE_BITS + first_bit % BYTE_BITS;
  const device uint* words = reinterpret_cast<const device uint*>(codes) + byte / 4;
  const uint high = words[0];
  const uint middle = words[1];
  const bool window_past_64_bits = shift + STATE_BITS + 3 * transition_bits > 64;
  const uint low = transition_bits == 7 ? words[1 + uint(window_past_64_bits)] : 0u;
  const uint top = (high << shift) | ((middle >> 1) >> (31 - shift));
  const uint bottom = (middle << shift) | ((low >> 1) >> (31 - shift));
  const uint state_mask = (1u << STATE_BITS) - 1;
  return uint4(
      state_weights(top >> STATE_BITS),
      state_weights((top >> (STATE_BITS - transition_bits)) & state_mask),
      state_weights((top >> (STATE_BITS - 2 * transition_bits)) & state_mask),
      state_weights(((top << (3 * transition_bits - STATE_BITS)) | (bottom >> (48 - 3 * transition_bits))) & state_mask)
  );
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
