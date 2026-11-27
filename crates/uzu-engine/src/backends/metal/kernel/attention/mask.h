#pragma once

#include "../common/defines.h"
#include "kv_layout.h"
#include "../generated/ring.h"
#include "../generated/trie.h"

using namespace uzu::ring;
using namespace uzu::trie;
using uzu::kv_layout::NO_KEY_SLOT;

// A key no query sees (unfilled, past the end); also the width of "no window".
constexpr constant uint INVALID_POSITION = 0x7FFFFFFFu;

struct KeyWindow {
  uint behind;
  uint span;
  bool is_sliding_window;

  METAL_FUNC static KeyWindow make(bool is_causal, bool is_sliding_window, const constant uint& sliding_window_size) {
    const uint window = is_sliding_window ? sliding_window_size : INVALID_POSITION;
    // Window 0 wraps: sees nothing.
    const uint behind = is_causal ? window - 1u : window / 2u;
    const uint ahead = is_causal ? 0u : window / 2u;
    return {behind, behind + ahead + 1u, is_sliding_window};
  }

  METAL_FUNC uint oldest_visible(uint prefix_length, uint query_position) const {
    return prefix_length - behind + query_position;
  }

  METAL_FUNC uint tile_oldest_visible(uint prefix_length, uint first_query) const {
    return is_sliding_window ? max(prefix_length + first_query, behind) - behind : 0u;
  }

  METAL_FUNC bool sees(uint oldest_visible, uint key_position) const { return key_position - oldest_visible < span; }
};

METAL_FUNC uint position_of_key(uint key, uint key_length, uint prefix_length, uint ring_offset) {
  if (key >= key_length) {
    return INVALID_POSITION;
  }
  if (key >= prefix_length) {
    return key;
  }
  return key >= ring_offset ? key - ring_offset : key + prefix_length - ring_offset;
}

METAL_FUNC uint2 tile_query_range(uint first_row, uint last_row, uint q_len, bool is_trie) {
  const bool one_head = first_row / q_len == last_row / q_len;
  return uint2(one_head && !is_trie ? first_row % q_len : 0u, one_head ? last_row % q_len : q_len - 1u);
}

struct RowRange {
  uint packed; // first | last << 16

  METAL_FUNC static RowRange make(uint first, uint last) { return {first | last << 16}; }
  METAL_FUNC static RowRange all() { return make(0u, metal::numeric_limits<ushort>::max()); }
  METAL_FUNC uint first() const thread { return packed & 0xFFFFu; }
  METAL_FUNC uint last() const thread { return packed >> 16; }
  METAL_FUNC bool contains(uint row) const thread { return row >= first() && row <= last(); }
};

struct VisibleKeyBlocks {
  uint ring_blocks;
  uint first_ring_block;
  uint visible_ring_blocks;
  uint first_suffix_block;
  uint causal_blocks;

  template <uint BLOCK_KEYS>
  METAL_FUNC static VisibleKeyBlocks make(
      const thread KeyWindow& window,
      bool is_causal,
      uint prefix_length,
      uint ring_offset,
      bool unrotated,
      uint2 query_range
  ) {
    const uint oldest_visible = window.tile_oldest_visible(prefix_length, query_range.x);
    const uint ring_blocks = prefix_length / BLOCK_KEYS;
    const uint visible_ring_keys = oldest_visible < prefix_length ? prefix_length - oldest_visible : 0u;
    const uint shifted = oldest_visible + ring_offset;
    const uint oldest_slot = shifted >= prefix_length ? shifted - prefix_length : shifted;
    const uint oldest_block = oldest_slot / BLOCK_KEYS;
    const bool oldest_in_partial_block = oldest_block >= ring_blocks;
    const uint first_ring_block = oldest_in_partial_block ? 0u : oldest_block;
    const uint visible_ring_blocks =
        unrotated || visible_ring_keys == 0u
            ? 0u
            : min(ring_blocks, div_ceil(oldest_slot % BLOCK_KEYS + visible_ring_keys, BLOCK_KEYS));
    VisibleKeyBlocks blocks = {
        .ring_blocks = ring_blocks,
        .first_ring_block = first_ring_block,
        .visible_ring_blocks = visible_ring_blocks,
        .first_suffix_block = unrotated ? 0u : max(oldest_visible, prefix_length) / BLOCK_KEYS
    };
    blocks.causal_blocks =
        is_causal ? blocks.count_until(div_ceil(prefix_length + query_range.y + 1u, BLOCK_KEYS)) : ~0u;
    return blocks;
  }

  METAL_FUNC uint count_until(uint end_block) const thread {
    return visible_ring_blocks + end_block - first_suffix_block;
  }

  METAL_FUNC uint block_at(uint i) const thread {
    const uint ring_block = first_ring_block + i;
    return i < visible_ring_blocks ? (ring_block >= ring_blocks ? ring_block - ring_blocks : ring_block)
                                   : first_suffix_block + i - visible_ring_blocks;
  }

  METAL_FUNC uint2 split(uint total_blocks, uint split, uint num_splits) const thread {
    const uint first = split * total_blocks / num_splits;
    return uint2(first, max(first, min((split + 1u) * total_blocks / num_splits, causal_blocks)));
  }
};

// The incomplete block is always attended independently of top-k.
struct PartialBlock {
  const device int* token_rows;
  uint token_count;

  METAL_FUNC uint slot(uint key) const { return key < token_count ? uint(token_rows[key]) : uint(NO_KEY_SLOT); }
};

// KeyBlocks packs per-query block IDs before the shared draft-member rows.
struct SparseKeys {
  const device int* block_ids;
  const device TrieNode* trie;
  const device int* draft_token_rows;
  uint compression_ratio;
  uint block_keys;
  uint contiguous_blocks;
  uint prefix_length;
  uint suffix_length;
  PartialBlock partial_block;
  uint query_position;
  bool is_trie;

  METAL_FUNC SparseKeys(
      const device int* selected_blocks,
      const uint blocks_per_query,
      const uint compression_ratio,
      const device TrieNode* trie,
      const uint trie_nodes,
      const uint prefix_length,
      const uint query_position,
      const bool is_trie
  )
      : block_ids(selected_blocks + query_position * blocks_per_query), trie(trie),
        draft_token_rows(selected_blocks + trie_nodes * blocks_per_query), compression_ratio(compression_ratio),
        block_keys(blocks_per_query * compression_ratio), prefix_length(prefix_length), suffix_length(trie_nodes),
        query_position(query_position), is_trie(is_trie) {
    const uint tokens = prefix_length + (is_trie ? trie[query_position].height : query_position) + 1u;
    contiguous_blocks = (is_trie ? prefix_length : tokens) / compression_ratio;
    partial_block.token_count = tokens % compression_ratio;
    partial_block.token_rows =
        draft_token_rows + query_position * compression_ratio + compression_ratio - partial_block.token_count;
  }

  METAL_FUNC uint key_count() const { return block_keys + partial_block.token_count; }

  METAL_FUNC uint slot(const uint key) const {
    if (key < block_keys) {
      const uint block = uint(block_ids[key / compression_ratio]);
      if (block < contiguous_blocks)
        return block * compression_ratio + key % compression_ratio;
      if (is_trie && block - contiguous_blocks < suffix_length) {
        const uint node_index = block - contiguous_blocks;
        const auto node = trie[node_index];
        if ((prefix_length + node.height + 1) % compression_ratio == 0 && query_position >= node.trie_start &&
            query_position <= node.trie_end)
          return uint(draft_token_rows[node_index * compression_ratio + key % compression_ratio]);
      }
      return uint(NO_KEY_SLOT);
    }
    return partial_block.slot(key - block_keys);
  }
};

METAL_FUNC uint slot_address(const uint slot) { return min(slot, ~slot); }
