#pragma once

#include <metal_stdlib>
#include "../common/defines.h"
#include "../generated/kv_layout.h"

// K: [block][head][fragment][lane][LANE_BYTES].
// V: [block][head][slice][chunk group][lane][LANE_BYTES].
// Unwritten slots have nonpositive scales.

namespace uzu::kv_layout {

// Lane bits: row[2], group[1], row[1], row[0], group[0].
METAL_FUNC uint mma_lane(const uint row_in_group, const uint channel_group) {
  return ((row_in_group >> 2) << 4) | ((channel_group >> 1) << 3) | (((row_in_group >> 1) & 1u) << 2) |
         ((row_in_group & 1u) << 1) | (channel_group & 1u);
}

struct Int8KVShape {
  uint head_dim;
  uint num_kv_heads;

  METAL_FUNC size_t block_region(const uint block, const uint kv_head) const {
    return (size_t(block) * num_kv_heads + kv_head) * (kv_layout::TILE_KEYS * head_dim);
  }

  METAL_FUNC size_t scale_index(const uint key, const uint kv_head) const {
    return (size_t(key / kv_layout::TILE_KEYS) * num_kv_heads + kv_head) * kv_layout::TILE_KEYS +
           key % kv_layout::TILE_KEYS;
  }

  METAL_FUNC size_t k_code_index(const uint key, const uint kv_head, const uint channel) const {
    const uint row = key % kv_layout::TILE_KEYS;
    const uint lane = mma_lane(row % kv_layout::GROUP_ROWS, channel % kv_layout::TILE_DIMS / kv_layout::LANE_COLS);
    const uint lane_offset = row / kv_layout::GROUP_ROWS * kv_layout::LANE_COLS + channel % kv_layout::LANE_COLS;
    return block_region(key / kv_layout::TILE_KEYS, kv_head) +
           channel / kv_layout::TILE_DIMS * kv_layout::FRAGMENT_BYTES + lane * kv_layout::LANE_BYTES + lane_offset;
  }

  METAL_FUNC size_t v_code_index(const uint key, const uint kv_head, const uint channel) const {
    const uint row = key % kv_layout::TILE_KEYS;
    const uint slice_channel = channel % kv_layout::SLICE_DIMS;
    const uint tile = slice_channel / kv_layout::TILE_DIMS;
    const uint tiles = kv_layout::SLICE_DIMS / kv_layout::TILE_DIMS;
    const uint half_rows = kv_layout::TILE_KEYS / 2;
    const uint lane =
        mma_lane(row % kv_layout::GROUP_ROWS, slice_channel % kv_layout::TILE_DIMS / kv_layout::LANE_COLS);
    const uint pair = tiles * (channel / kv_layout::SLICE_DIMS) + tiles / 2 * (row / half_rows) + tile / 2;
    const uint lane_offset = tile % 2 * (kv_layout::LANE_BYTES / 2) +
                             row % half_rows / kv_layout::GROUP_ROWS * kv_layout::LANE_COLS +
                             channel % kv_layout::LANE_COLS;
    return block_region(key / kv_layout::TILE_KEYS, kv_head) + pair * kv_layout::FRAGMENT_BYTES +
           lane * kv_layout::LANE_BYTES + lane_offset;
  }
};

METAL_FUNC float rotation_sign(const uint channel) {
  const uint word = kv_layout::ROTATION_SIGN_BITS[channel / kv_layout::WORD_BITS];
  const uint bit = channel % kv_layout::WORD_BITS;
  return (word >> bit) & 1u ? -1.0f : 1.0f;
}

} // namespace uzu::kv_layout
