#pragma once

#include <metal_stdlib>
#include "../common/defines.h"
#include "../generated/kv_layout.h"

namespace uzu::kv_layout {

// Lane bits: row[2], group[1], row[1], row[0], group[0].
METAL_FUNC uint mma_lane(const uint row_in_group, const uint channel_group) {
  return ((row_in_group >> 2) << 4) | ((channel_group >> 1) << 3) | (((row_in_group >> 1) & 1u) << 2) |
         ((row_in_group & 1u) << 1) | (channel_group & 1u);
}

struct KVShape {
  uint head_dim;
  uint num_kv_heads;
  KVLayout layout;

  METAL_FUNC size_t block_region(const uint block, const uint kv_head) const {
    return (size_t(block) * num_kv_heads + kv_head) * (kv_layout::TILE_KEYS * head_dim);
  }

  // Scales: [key block][head][key in block].
  METAL_FUNC uint scale_index(const uint key, const uint kv_head) const {
    return (key / kv_layout::TILE_KEYS * num_kv_heads + kv_head) * kv_layout::TILE_KEYS + key % kv_layout::TILE_KEYS;
  }

  METAL_FUNC size_t k_code_index(const uint key, const uint kv_head, const uint channel) const {
    if (layout == KVLayout::RowMajor)
      return size_t(scale_index(key, kv_head)) * head_dim + channel;
    // K: [block][head][fragment][lane][LANE_BYTES].
    const uint row = key % kv_layout::TILE_KEYS;
    const uint lane = mma_lane(row % kv_layout::GROUP_ROWS, channel % kv_layout::TILE_DIMS / kv_layout::LANE_COLS);
    const uint lane_offset = row / kv_layout::GROUP_ROWS * kv_layout::LANE_COLS + channel % kv_layout::LANE_COLS;
    return block_region(key / kv_layout::TILE_KEYS, kv_head) +
           channel / kv_layout::TILE_DIMS * kv_layout::FRAGMENT_BYTES + lane * kv_layout::LANE_BYTES + lane_offset;
  }

  METAL_FUNC size_t v_code_index(const uint key, const uint kv_head, const uint channel) const {
    if (layout == KVLayout::RowMajor)
      return k_code_index(key, kv_head, channel);
    // V: [block][head][slice][chunk group][lane][LANE_BYTES].
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

  // k_code_index(key, head, channel) == k_key_offset(key, head) + k_channel_offset(channel) and likewise for V: the
  // channel part does not depend on the key, so a kernel that visits many keys computes it once.
  METAL_FUNC size_t k_key_offset(const uint key, const uint kv_head) const {
    if (layout == KVLayout::RowMajor)
      return size_t(scale_index(key, kv_head)) * head_dim;
    const uint row = key % kv_layout::TILE_KEYS;
    return block_region(key / kv_layout::TILE_KEYS, kv_head) +
           mma_lane(row % kv_layout::GROUP_ROWS, 0) * kv_layout::LANE_BYTES +
           row / kv_layout::GROUP_ROWS * kv_layout::LANE_COLS;
  }

  METAL_FUNC uint k_channel_offset(const uint channel) const {
    if (layout == KVLayout::RowMajor)
      return channel;
    return channel / kv_layout::TILE_DIMS * kv_layout::FRAGMENT_BYTES +
           mma_lane(0, channel % kv_layout::TILE_DIMS / kv_layout::LANE_COLS) * kv_layout::LANE_BYTES +
           channel % kv_layout::LANE_COLS;
  }

  METAL_FUNC size_t v_key_offset(const uint key, const uint kv_head) const {
    if (layout == KVLayout::RowMajor)
      return k_key_offset(key, kv_head);
    const uint row = key % kv_layout::TILE_KEYS;
    const uint tiles = kv_layout::SLICE_DIMS / kv_layout::TILE_DIMS;
    const uint half_rows = kv_layout::TILE_KEYS / 2;
    return block_region(key / kv_layout::TILE_KEYS, kv_head) +
           tiles / 2 * (row / half_rows) * kv_layout::FRAGMENT_BYTES +
           mma_lane(row % kv_layout::GROUP_ROWS, 0) * kv_layout::LANE_BYTES +
           row % half_rows / kv_layout::GROUP_ROWS * kv_layout::LANE_COLS;
  }

  METAL_FUNC uint v_channel_offset(const uint channel) const {
    if (layout == KVLayout::RowMajor)
      return channel;
    const uint slice_channel = channel % kv_layout::SLICE_DIMS;
    const uint tile = slice_channel / kv_layout::TILE_DIMS;
    const uint tiles = kv_layout::SLICE_DIMS / kv_layout::TILE_DIMS;
    return (tiles * (channel / kv_layout::SLICE_DIMS) + tile / 2) * kv_layout::FRAGMENT_BYTES +
           mma_lane(0, slice_channel % kv_layout::TILE_DIMS / kv_layout::LANE_COLS) * kv_layout::LANE_BYTES +
           tile % 2 * (kv_layout::LANE_BYTES / 2) + channel % kv_layout::LANE_COLS;
  }
};

METAL_FUNC float rotation_sign(const uint channel) {
  const uint word = kv_layout::ROTATION_SIGN_BITS[(channel / kv_layout::WORD_BITS) % 8];
  const uint bit = channel % kv_layout::WORD_BITS;
  return (word >> bit) & 1u ? -1.0f : 1.0f;
}

} // namespace uzu::kv_layout
