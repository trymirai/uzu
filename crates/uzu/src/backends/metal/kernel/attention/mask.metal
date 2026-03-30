#include "../generated/ring.h"
#include "../generated/trie.h"

using namespace uzu::ring;
using namespace uzu::trie;

bool should_use_key(
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
        use_key &=
            q_seq_idx >= key_trie.trie_start && q_seq_idx <= key_trie.trie_end;
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
      key_position =
          (prefix_length + i - ring_params.ring_offset) % prefix_length;
      // Ring also skips entries that aren't filled yet
      use_key &= key_position < ring_params.ring_length;
    } else {
      // Non-ring, position equals to the index in the kvcache buffer
      key_position = i;
    }
  }

  if (is_sliding_window) {
    if (is_causal) {
      use_key &= key_position <= query_position &&
                 (query_position - key_position) < sliding_window_size;
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
