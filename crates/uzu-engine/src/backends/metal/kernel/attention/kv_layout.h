#pragma once

#include <metal_stdlib>
#include "../common/defines.h"
#include "../generated/kv_layout.h"

namespace uzu::kv_layout {

// Lane bits: row[2], group[1], row[1], row[0], group[0].
METAL_FUNC uint lane_row(const uint lane) {
  return ((lane >> 4) << 2) | (((lane >> 2) & 1u) << 1) | ((lane >> 1) & 1u);
}

METAL_FUNC uint lane_group(const uint lane) { return (((lane >> 3) & 1u) << 1) | (lane & 1u); }

constexpr constant uint LANE_KEYS = TILE_KEYS / GROUP_ROWS;
static_assert(COMPRESSION_RATIO == LANE_KEYS, "a sparse block is one lane group of keys");
constexpr constant uint PANEL_TILES = 2;
constexpr constant uint PAIR_KEYS = 2;
constexpr constant uint SLICE_PANELS = SLICE_DIMS / (PANEL_TILES * TILE_DIMS);
constexpr constant uint TILE_LANE_BYTES = LANE_BYTES / PANEL_TILES;
constexpr constant uint BLOCK_FRAGMENT_BYTES = TILE_DIMS / LANE_COLS * LANE_BYTES;

METAL_FUNC uint lane_first_key(const uint lane) { return lane_row(lane) * LANE_KEYS; }

METAL_FUNC uint lane_channel(const uint lane) { return lane_group(lane) * LANE_COLS; }

METAL_FUNC uint column_key(const uint column) { return LANE_KEYS * (column % GROUP_ROWS) + column / GROUP_ROWS; }

struct KVShape {
  uint head_dim;
  uint num_kv_heads;
  KVLayout layout;

  METAL_FUNC uint block_stride() const { return kv_layout::TILE_KEYS * num_kv_heads * head_dim; }

  METAL_FUNC uint scale_block_stride() const { return kv_layout::TILE_KEYS * num_kv_heads; }

  METAL_FUNC size_t block_region(const uint block, const uint kv_head) const {
    return (size_t(block) * num_kv_heads + kv_head) * (kv_layout::TILE_KEYS * head_dim);
  }

  // Scales: [key block][head][key in block], in MMA column order when MxuTiled.
  METAL_FUNC uint scale_index(const uint key, const uint kv_head) const {
    if (layout == KVLayout::MxuTiled)
      return (key / kv_layout::TILE_KEYS * num_kv_heads + kv_head) * kv_layout::TILE_KEYS +
             kv_layout::GROUP_ROWS * (key % kv_layout::LANE_KEYS) + key % kv_layout::TILE_KEYS / kv_layout::LANE_KEYS;
    return (key / kv_layout::TILE_KEYS * num_kv_heads + kv_head) * kv_layout::TILE_KEYS + key % kv_layout::TILE_KEYS;
  }

  METAL_FUNC size_t row_code_index(const uint row, const uint channel) const {
    return size_t(row) * head_dim + channel;
  }

  METAL_FUNC size_t k_code_index(const uint key, const uint kv_head, const uint channel) const {
    if (layout == KVLayout::RowMajor)
      return size_t(scale_index(key, kv_head)) * head_dim + channel;
    // K: [block][head][key block][fragment][channel group][key][channel].
    const uint row = key % kv_layout::TILE_KEYS;
    return block_region(key / kv_layout::TILE_KEYS, kv_head) +
           row / kv_layout::LANE_KEYS * (kv_layout::LANE_KEYS * head_dim) +
           channel / kv_layout::TILE_DIMS * kv_layout::BLOCK_FRAGMENT_BYTES +
           channel % kv_layout::TILE_DIMS / kv_layout::LANE_COLS * kv_layout::LANE_BYTES +
           row % kv_layout::LANE_KEYS * kv_layout::LANE_COLS + channel % kv_layout::LANE_COLS;
  }

  METAL_FUNC size_t v_code_index(const uint key, const uint kv_head, const uint channel) const {
    if (layout == KVLayout::RowMajor)
      return k_code_index(key, kv_head, channel);
    // V: [block][head][key block][slice][key pair][panel][channel group][tile][key][channel].
    const uint row = key % kv_layout::TILE_KEYS;
    const uint slice_channel = channel % kv_layout::SLICE_DIMS;
    const uint tile = slice_channel / kv_layout::TILE_DIMS;
    const uint key_in_block = row % kv_layout::LANE_KEYS;
    return block_region(key / kv_layout::TILE_KEYS, kv_head) +
           row / kv_layout::LANE_KEYS * (kv_layout::LANE_KEYS * head_dim) +
           channel / kv_layout::SLICE_DIMS * (kv_layout::LANE_KEYS * kv_layout::SLICE_DIMS) +
           (key_in_block / kv_layout::PAIR_KEYS * kv_layout::SLICE_PANELS + tile / kv_layout::PANEL_TILES) *
               kv_layout::BLOCK_FRAGMENT_BYTES +
           slice_channel % kv_layout::TILE_DIMS / kv_layout::LANE_COLS * kv_layout::LANE_BYTES +
           tile % kv_layout::PANEL_TILES * kv_layout::TILE_LANE_BYTES +
           key_in_block % kv_layout::PAIR_KEYS * kv_layout::LANE_COLS + channel % kv_layout::LANE_COLS;
  }
};

METAL_FUNC float rotation_sign(const uint channel) {
  const uint word = kv_layout::ROTATION_SIGN_BITS[(channel / kv_layout::WORD_BITS) % 8];
  const uint bit = channel % kv_layout::WORD_BITS;
  return (word >> bit) & 1u ? -1.0f : 1.0f;
}

} // namespace uzu::kv_layout
