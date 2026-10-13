#pragma once

#include <metal_stdlib>
#include "../common/defines.h"

// clang-format off
//   k_codes  int8   [key block][kv head][fragment][lane][16 B]                     one fragment = 32 keys x 16 channels
//   v_codes  uint8  [key block][kv head][channel slice][chunk group][lane][16 B]   (code + 128), the P x V operand
//   k_scales bf16   [kv head][key]                                                 stride = padded capacity
//   v_scales bf16   [kv head][key]                                                 stride = padded capacity
// clang-format on

namespace uzu::kv_layout {

UZU_CONST uint TILE_KEYS = 32;
UZU_CONST uint TILE_DIMS = 16;
UZU_CONST uint FRAGMENT_BYTES = TILE_KEYS * TILE_DIMS;
UZU_CONST uint KV_SLICE_DIMS = 64;
UZU_CONST int KV_V_CODE_BIAS = 128;

/// The lane whose relaxed-MXU `get_position` is `(4 * channel_group, row_in_group)` (inverse of the fragment lane map).
METAL_FUNC uint mma_lane(const uint row_in_group, const uint channel_group) {
  return 16u * (row_in_group >> 2) + 8u * (channel_group >> 1) + 4u * ((row_in_group >> 1) & 1u) +
         2u * (row_in_group & 1u) + (channel_group & 1u);
}

struct Int8KVShape {
  uint head_dim;
  uint num_kv_heads;

  METAL_FUNC size_t block_region(const uint block, const uint kv_head) const {
    return (size_t(block) * num_kv_heads + kv_head) * (TILE_KEYS * head_dim);
  }

  METAL_FUNC size_t k_code_index(const uint key, const uint kv_head, const uint channel) const {
    const uint row = key % TILE_KEYS;
    const uint lane = mma_lane(row % 8, channel % TILE_DIMS / 4);
    return block_region(key / TILE_KEYS, kv_head) + channel / TILE_DIMS * FRAGMENT_BYTES + lane * 16 + row / 8 * 4 +
           channel % 4;
  }

  METAL_FUNC size_t v_code_index(const uint key, const uint kv_head, const uint channel) const {
    const uint row = key % TILE_KEYS;
    const uint slice_channel = channel % KV_SLICE_DIMS;
    const uint fragment_in_slice = slice_channel / TILE_DIMS;
    const uint lane = mma_lane(row % 8, slice_channel % TILE_DIMS / 4);
    // Each lane chunk holds two fragments (8 bytes each: 4 channels of two key rows) for 16 keys.
    const uint fragment_pair = 4 * (channel / KV_SLICE_DIMS) + 2 * (row / 16) + fragment_in_slice / 2;
    return block_region(key / TILE_KEYS, kv_head) + fragment_pair * FRAGMENT_BYTES + lane * 16 +
           fragment_in_slice % 2 * 8 + row % 16 / 8 * 4 + channel % 4;
  }
};

METAL_FUNC size_t per_token_index(const uint key, const uint kv_head, const uint padded_capacity) {
  return size_t(kv_head) * padded_capacity + key;
}

} // namespace uzu::kv_layout
