use crate::backends::common::gpu_types::{ring::RingParams, trie::TrieNode};

pub fn should_use_key(
    ring_params: Option<RingParams>,
    trie: Option<*const TrieNode>,
    sliding_window_size: Option<u32>,
    q_seq_idx: u32,
    prefix_length: u32,
    suffix_position: u32,
    query_position: u32,
    i: u32,
    is_causal: bool,
) -> bool {
    let mut use_key = true;
    let key_position;

    if i >= prefix_length {
        // We're in the suffix
        let key_position_in_suffix = i - prefix_length;

        if let Some(trie) = trie {
            // Trie, get position from the flat trie buffer
            let trie_node = unsafe { &*trie.add(key_position_in_suffix as usize) };

            key_position = suffix_position + trie_node.height;

            if is_causal {
                use_key &= q_seq_idx >= trie_node.trie_start && q_seq_idx <= trie_node.trie_end;
            }
        } else {
            // Non-trie, position equals index in the suffix
            key_position = suffix_position + key_position_in_suffix;

            if is_causal {
                use_key &= key_position_in_suffix <= q_seq_idx;
            }
        }
    } else {
        // We're in the prefix
        if let Some(ring_params) = ring_params {
            // Ring, calculate position in the ring
            key_position = (prefix_length + i - ring_params.ring_offset) % prefix_length;
            // Ring also skips entries that aren't filled yet
            use_key &= key_position < ring_params.ring_length;
        } else {
            // Non-ring, position equals to the index in the kvcache buffer
            key_position = i;
        }
    }

    if let Some(sliding_window_size) = sliding_window_size {
        if is_causal {
            use_key &= key_position <= query_position && (query_position - key_position) < sliding_window_size;
        } else if key_position <= query_position {
            use_key &= (query_position - key_position) <= sliding_window_size / 2;
        } else {
            use_key &= (key_position - query_position) <= sliding_window_size / 2;
        }
    }

    use_key
}
