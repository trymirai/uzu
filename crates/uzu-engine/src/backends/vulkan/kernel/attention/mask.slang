import definitions;
import generated.ring;
import generated.trie;

// The keys the queries of a suffix use, Metal's should_use_key in its unsigned arithmetic: suffix keys at their trie
// height or suffix index, by ancestry or index order under causality; ring prefix keys at their ring position, filled
// entries only; then the sliding window.
public struct AttentionMask {
  public RingParams ring_params;
  public Ptr<TrieNode, Access.Read> trie;
  public uint sliding_window_size;
  public uint prefix_length;
  public uint suffix_position;
  public bool is_kv_cache_ring;
  public bool is_causal;
  public bool is_trie;
  public bool is_sliding_window;

  public __init(
    uint sequence_length,
    uint suffix_length,
    RingParams ring_params,
    Ptr<TrieNode, Access.Read> trie,
    uint sliding_window_size,
    bool is_kv_cache_ring,
    bool is_causal,
    bool is_trie,
    bool is_sliding_window
  ) {
    this.ring_params = ring_params;
    this.trie = trie;
    this.sliding_window_size = sliding_window_size;
    this.prefix_length = sequence_length - suffix_length;
    this.suffix_position = is_kv_cache_ring ? ring_params.ring_length : prefix_length;
    this.is_kv_cache_ring = is_kv_cache_ring;
    this.is_causal = is_causal;
    this.is_trie = is_trie;
    this.is_sliding_window = is_sliding_window;
  }

  // The position of suffix query `query`: its trie height or its index, past the prefix.
  public uint query_position(uint query) {
    return suffix_position + (is_trie ? trie[query].height : query);
  }

  // Whether suffix query `query`, at `query_position`, uses `key`.
  public bool admits(uint key, uint query, uint query_position) {
    if (!is_trie && !is_kv_cache_ring && !is_sliding_window) {
      return !is_causal || key <= prefix_length + query;
    }
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
}
