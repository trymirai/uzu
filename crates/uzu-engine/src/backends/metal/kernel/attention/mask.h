#pragma once

#include "kv_layout.h"
#include "../generated/ring.h"
#include "../generated/trie.h"

using namespace uzu::ring;
using namespace uzu::trie;
using uzu::kv_layout::NO_KEY_SLOT;

static inline bool should_use_key(
    const constant RingParams& ring_params,
    const device TrieNode* trie,
    const constant uint& sliding_window_size,
    const uint q_seq_idx,
    const uint prefix_length,
    const uint suffix_position,
    const uint query_position,
    const uint i,
    const bool is_kv_cache_ring,
    const bool is_causal,
    const bool is_trie,
    const bool is_sliding_window
) {
  bool use_key = true;

  uint key_position;

  if (i >= prefix_length) {
    // We're in the suffix
    uint key_position_in_suffix = i - prefix_length;

    if (is_trie) {
      // Trie, get position from the flat trie buffer
      TrieNode key_trie = trie[key_position_in_suffix];

      key_position = suffix_position + key_trie.height;

      if (is_causal) {
        use_key &= q_seq_idx >= key_trie.trie_start && q_seq_idx <= key_trie.trie_end;
      }
    } else {
      // Non-trie, position equals index in the suffix
      key_position = suffix_position + key_position_in_suffix;

      if (is_causal) {
        use_key &= key_position_in_suffix <= q_seq_idx;
      }
    }

  } else {
    // We're in the prefix
    if (is_kv_cache_ring) {
      // Ring, calculate position in the ring
      key_position = (prefix_length + i - ring_params.ring_offset) % prefix_length;
      // Ring also skips entries that aren't filled yet
      use_key &= key_position < ring_params.ring_length;
    } else {
      // Non-ring, position equals to the index in the kvcache buffer
      key_position = i;
    }
  }

  if (is_sliding_window) {
    if (is_causal) {
      use_key &= key_position <= query_position && (query_position - key_position) < sliding_window_size;
    } else {
      if (key_position <= query_position) {
        use_key &= (query_position - key_position) <= (sliding_window_size / 2);
      } else {
        use_key &= (key_position - query_position) <= (sliding_window_size / 2);
      }
    }
  }

  return use_key;
}

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
  const device int* block_token_rows;
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
        block_token_rows(selected_blocks + trie_nodes * blocks_per_query), compression_ratio(compression_ratio),
        block_keys(blocks_per_query * compression_ratio), prefix_length(prefix_length), suffix_length(trie_nodes),
        query_position(query_position), is_trie(is_trie) {
    const uint tokens = prefix_length + (is_trie ? trie[query_position].height : query_position) + 1u;
    contiguous_blocks = (is_trie ? prefix_length : tokens) / compression_ratio;
    partial_block.token_count = tokens % compression_ratio;
    partial_block.token_rows =
        block_token_rows + query_position * compression_ratio + compression_ratio - partial_block.token_count;
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
          return uint(block_token_rows[node_index * compression_ratio + key % compression_ratio]);
      }
      return uint(NO_KEY_SLOT);
    }
    return partial_block.slot(key - block_keys);
  }
};

METAL_FUNC uint slot_address(const uint slot) { return min(slot, ~slot); }
