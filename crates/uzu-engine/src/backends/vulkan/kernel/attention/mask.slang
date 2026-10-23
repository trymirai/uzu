import definitions;
import generated.ring;
import generated.trie;

// The CPU should_use_key in its unsigned arithmetic: suffix keys at their trie height or suffix index, by ancestry or
// index order under causality; ring prefix keys at their ring position, filled entries only; then the sliding window.
public bool attention_use_key(
  uint key,
  uint query,
  uint prefix_length,
  uint suffix_position,
  uint query_position,
  RingParams ring_params,
  Ptr<TrieNode, Access.Read> trie,
  uint sliding_window_size,
  bool is_kv_cache_ring,
  bool is_causal,
  bool is_trie,
  bool is_sliding_window
) {
  var use_key = true;
  uint key_position;
  if (key >= prefix_length) {
    let in_suffix = key - prefix_length;
    if (is_trie) {
      let node = trie[in_suffix];
      key_position = suffix_position + node.height;
      if (is_causal) {
        use_key = query >= node.trie_start && query <= node.trie_end;
      }
    } else {
      key_position = suffix_position + in_suffix;
      if (is_causal) {
        use_key = in_suffix <= query;
      }
    }
  } else if (is_kv_cache_ring) {
    key_position = (prefix_length + key - ring_params.ring_offset) % prefix_length;
    use_key = key_position < ring_params.ring_length;
  } else {
    key_position = key;
  }
  if (is_sliding_window) {
    if (is_causal) {
      use_key = use_key && key_position <= query_position && query_position - key_position < sliding_window_size;
    } else if (key_position <= query_position) {
      use_key = use_key && query_position - key_position <= sliding_window_size / 2;
    } else {
      use_key = use_key && key_position - query_position <= sliding_window_size / 2;
    }
  }
  return use_key;
}
