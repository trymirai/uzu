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

struct SparseKeys {
  const device int* blocks;
  const device TrieNode* trie;
  uint compression_ratio;
  uint block_keys;
  uint selectable_blocks;
  uint prefix_length;
  uint key_end;
  uint query_position;
  bool is_trie;

  METAL_FUNC SparseKeys(
      const device int* query_blocks,
      const uint blocks_per_query,
      const uint compression_ratio,
      const device TrieNode* trie,
      const uint trie_nodes,
      const uint prefix_length,
      const uint query_position,
      const bool is_trie
  )
      : blocks(query_blocks + query_position * blocks_per_query), trie(trie), compression_ratio(compression_ratio),
        block_keys(blocks_per_query * compression_ratio), prefix_length(prefix_length),
        key_end(is_trie ? prefix_length + trie_nodes : prefix_length + query_position + 1u),
        query_position(query_position), is_trie(is_trie) {
    selectable_blocks = (is_trie ? prefix_length : key_end) / compression_ratio;
  }

  METAL_FUNC uint key_count() const { return block_keys + key_end - selectable_blocks * compression_ratio; }

  METAL_FUNC uint slot(const uint key) const {
    if (key < block_keys) {
      const uint block = uint(blocks[key / compression_ratio]);
      return block < selectable_blocks ? block * compression_ratio + key % compression_ratio : uint(NO_KEY_SLOT);
    }
    const uint tail_slot = selectable_blocks * compression_ratio + key - block_keys;
    if (tail_slot >= key_end) {
      return uint(NO_KEY_SLOT);
    }
    if (is_trie && tail_slot >= prefix_length) {
      const TrieNode node = trie[tail_slot - prefix_length];
      if (query_position < node.trie_start || query_position > node.trie_end) {
        return ~tail_slot;
      }
    }
    return tail_slot;
  }
};

METAL_FUNC uint slot_address(const uint slot) { return min(slot, ~slot); }
