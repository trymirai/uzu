use uzu_engine_macros::kernel;

use crate::backends::common::gpu_types::trie::TrieNode;

#[kernel(VerifyTrie)]
pub fn verify_trie(
    trie_nodes: *const TrieNode,
    parents: *const i32,
    token_ids: *const u32,
    sampled_tokens: *const u32,
    output_norm: *const u8,
    accepted_indices: *mut u32,
    accepted_input_tokens: *mut u32,
    accepted_output_tokens: *mut u32,
    accepted_output_norm: *mut u8,
    num_nodes: u32,
    capacity: u32,
    norm_row_bytes: u32,
) {
    let _ = parents;

    let num_nodes = num_nodes as usize;
    let capacity = capacity as usize;
    let norm_row_bytes = norm_row_bytes as usize;

    if num_nodes == 0 {
        return;
    }

    let trie_nodes = unsafe { std::slice::from_raw_parts(trie_nodes, num_nodes) };
    let token_ids = unsafe { std::slice::from_raw_parts(token_ids, num_nodes) };
    let sampled_tokens = unsafe { std::slice::from_raw_parts(sampled_tokens, num_nodes) };

    let find_child = |current: usize, sampled: u32| -> Option<usize> {
        let end = trie_nodes[current].trie_end as usize;
        let mut child = current + 1;
        while child <= end {
            if token_ids[child] == sampled {
                return Some(child);
            }
            child = trie_nodes[child].trie_end as usize + 1;
        }
        None
    };

    let mut len = 0;
    let mut current = 0;
    loop {
        len += 1;
        match find_child(current, sampled_tokens[current]) {
            Some(child) => current = child,
            None => break,
        }
    }
    let last = current;

    assert!(len < capacity, "accepted path is longer than the output buffers");

    unsafe {
        *accepted_indices.add(0) = len as u32;
        *accepted_input_tokens.add(0) = len as u32;
        *accepted_output_tokens.add(0) = len as u32;

        let mut entry = 0;
        let mut current = 0;
        loop {
            let position = len - entry;
            *accepted_indices.add(position) = current as u32;
            *accepted_input_tokens.add(position) = token_ids[current];
            *accepted_output_tokens.add(position) = sampled_tokens[current];
            entry += 1;
            if entry == len {
                break;
            }
            current = find_child(current, sampled_tokens[current]).unwrap();
        }

        let norm_row = std::slice::from_raw_parts(output_norm.add(last * norm_row_bytes), norm_row_bytes);
        let accepted_norm_row = std::slice::from_raw_parts_mut(accepted_output_norm, norm_row_bytes);
        accepted_norm_row.copy_from_slice(norm_row);
    }
}
