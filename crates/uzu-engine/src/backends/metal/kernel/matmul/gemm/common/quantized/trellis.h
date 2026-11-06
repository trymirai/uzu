#pragma once

#include <metal_stdlib>

#include "../../../../common/defines.h"
#include "../../../../generated/gemm.h"

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
UZU_CONST uint STATE_BITS = 16, BYTE_BITS = 8, WORD_BYTES = 4, WORD_BITS = 32, LANE_COLUMNS = 16, ROW_ALIGNMENT = 16;

UZU_CONST uint HASH_MULTIPLIER = 0xCFCCB83Fu, HASH_INCREMENT = 0x584B4AA3u;
UZU_CONST uint HASH_AVALANCHE_MULTIPLIER = 0x85EBCA6Bu, HASH_FOLD_BITS = 16;

UZU_CONST uint TWO_BIT_FIELD_SHIFT = 2, NIBBLE_SHIFT = 4, FIELD_SUM_SHIFT = 3;
UZU_CONST uint TWO_BIT_FIELDS_MASK = 0x33333333u, BYTE_NIBBLES_MASK = 0x0F0F0F0Fu;
UZU_CONST uint FIELD_PAIR_SUM_MULTIPLIER = 3, DITHER_MULTIPLIER = 3;

// Row layout: restart rows pad to ROW_ALIGNMENT bytes, connected rows to a byte
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

// Tape 2: each block's bits reversed, LSB first; state i is the LE 16-bit field at bit B - 16 - i*T
static METAL_FUNC uint states_per_lane(const GemmTrellisFormat format) { return LANE_COLUMNS / format.vector_width; }

static METAL_FUNC uint states_per_window(const GemmTrellisFormat format) {
  const uint block_states = format.restart_columns / format.vector_width;
  return format.restart_columns == 0 ? states_per_lane(format) : min(states_per_lane(format), 1u << ctz(block_states));
}

static METAL_FUNC uint window_bits(const GemmTrellisFormat format) {
  return STATE_BITS + (states_per_window(format) - 1) * format.transition_bits;
}

// Window starts are multiples of a power of two bits, at least a byte (row starts are only byte aligned)
static METAL_FUNC uint max_window_bit_offset(const GemmTrellisFormat format) {
  const uint spacing = states_per_window(format) * format.transition_bits;
  uint unit = spacing;
  if (format.restart_columns != 0) {
    unit |= block_bits(format, format.restart_columns) | (spacing - format.transition_bits);
  }
  return WORD_BITS - min(BYTE_BITS, 1u << ctz(unit));
}

static METAL_FUNC uint state_field(const uint word0, const uint word1, const uint word2, const uint first_bit) {
  const uint word = first_bit / WORD_BITS, bit = first_bit % WORD_BITS;
  const uint low = word == 0 ? word0 : (word == 1 ? word1 : word2);
  uint value = low >> bit;
  if (bit + STATE_BITS > WORD_BITS) {
    value |= (word == 0 ? word1 : word2) << (WORD_BITS - bit);
  }
  return value & ((1u << STATE_BITS) - 1);
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

static METAL_FUNC uint state_pair_weights(const uint first_state, const uint second_state) {
  const uint first_hash = hash_without_final_fold(first_state);
  const uint second_hash = hash_without_final_fold(second_state);
  const uint low_halves = insert_bits(first_hash, second_hash, HASH_FOLD_BITS, HASH_FOLD_BITS);
  const uint high_halves = insert_bits(second_hash, first_hash >> HASH_FOLD_BITS, 0, HASH_FOLD_BITS);
  return biased_weights(low_halves ^ high_halves);
}

// Biased weights of the 16 columns from code_column, four per word
static METAL_FUNC uint4 decode_biased_weight_quad(
    const GemmTrellisFormat format,
    const device uchar* codes,
    const uint row_start_byte,
    const uint code_column,
    const uint row_columns
) {
  const uint transition_bits = format.transition_bits;
  const uint window_states = states_per_window(format);
  const uint states_per_weight_word = WORD_BYTES / format.vector_width;
  const uint block_columns = format.restart_columns != 0 ? format.restart_columns : row_columns;
  const uint block_size = block_bits(format, block_columns);
  const uint bits_in_window = window_bits(format);
  const bool third_word_possible = max_window_bit_offset(format) + bits_in_window > 2 * WORD_BITS;
  uint4 weights;
  METAL_PRAGMA_UNROLL
  for (uint window = 0; window < states_per_lane(format) / window_states; ++window) {
    const uint column = code_column + window * window_states * format.vector_width;
    const uint block = format.restart_columns != 0 ? column / block_columns : 0;
    const uint state_in_block = column % block_columns / format.vector_width;
    const uint first_bit =
        block * block_size + block_size - STATE_BITS - (state_in_block + window_states - 1) * transition_bits;
    const uint byte = row_start_byte + first_bit / BYTE_BITS;
    uint3 words(0u);
    uint bit_offset;
    if (format.restart_columns != 0) {
      bit_offset = byte % WORD_BYTES * BYTE_BITS + first_bit % BYTE_BITS;
      const device uint* aligned_words = reinterpret_cast<const device uint*>(codes) + byte / WORD_BYTES;
      const bool word_pair_fits = format.vector_width == 4 && transition_bits == 8 && format.restart_columns == 64;
      words.x = aligned_words[0];
      // Reuse an existing word to avoid conditional loads.
      const uint second_index = word_pair_fits || bit_offset + bits_in_window > WORD_BITS ? 1u : 0u;
      words.y = aligned_words[second_index];
      const uint third_index = bit_offset + bits_in_window > 2 * WORD_BITS ? 2u : (word_pair_fits ? 1u : 0u);
      words.z = third_word_possible ? aligned_words[third_index] : 0u;
    } else {
      const uint max_bit_offset = format.vector_width == 2 || transition_bits % 2 == 0 ? 0u : 4u;
      bit_offset = max_bit_offset == 0 ? 0u : first_bit % BYTE_BITS;
      const uint min_byte_count = div_ceil(bits_in_window, BYTE_BITS);
      const uint full_word_bytes = min_byte_count / WORD_BYTES * WORD_BYTES;
      const uint max_byte_count = div_ceil(bits_in_window + max_bit_offset, BYTE_BITS);
      const device packed_uchar4* packed_words = reinterpret_cast<const device packed_uchar4*>(codes + byte);
      if (full_word_bytes >= WORD_BYTES) {
        words.x = as_type<uint>(packed_words[0]);
      }
      if (full_word_bytes >= 2 * WORD_BYTES) {
        words.y = as_type<uint>(packed_words[1]);
      }
      METAL_PRAGMA_UNROLL
      for (uint byte_index = full_word_bytes; byte_index < max_byte_count; ++byte_index) {
        if (byte_index * BYTE_BITS < bit_offset + bits_in_window) {
          const uint value = uint(codes[byte + byte_index]) << (byte_index % WORD_BYTES * BYTE_BITS);
          if (byte_index < WORD_BYTES) {
            words.x |= value;
          } else if (byte_index < 2 * WORD_BYTES) {
            words.y |= value;
          } else {
            words.z |= value;
          }
        }
      }
    }
    const uint word0 = (words.x >> bit_offset) | ((words.y << 1) << (WORD_BITS - 1 - bit_offset));
    const uint word1 = (words.y >> bit_offset) | ((words.z << 1) << (WORD_BITS - 1 - bit_offset));
    const uint word2 = words.z >> bit_offset;
    METAL_PRAGMA_UNROLL
    for (uint output = 0; output < window_states / states_per_weight_word; ++output) {
      const uint first_state_bit = (window_states - 1 - output * states_per_weight_word) * transition_bits;
      const uint first_state = state_field(word0, word1, word2, first_state_bit);
      weights[window * (window_states / states_per_weight_word) + output] =
          format.vector_width == 4
              ? biased_weights(final_fold(hash_without_final_fold(first_state)))
              : state_pair_weights(first_state, state_field(word0, word1, word2, first_state_bit - transition_bits));
    }
  }
  return weights;
}

} // namespace trellis

} // namespace gemm
} // namespace uzu
