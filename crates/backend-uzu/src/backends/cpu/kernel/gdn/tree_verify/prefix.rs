use proc_macros::kernel;

use crate::backends::common::gpu_types::trie::TrieNode;

#[kernel(BuildTreePrefix)]
pub fn build_tree_prefix(
    trie: *const TrieNode,
    log_decay: *const f32,
    prefix: *mut f32,
    batch_size: u32,
    tree_size: u32,
    value_heads: u32,
) {
    let batch_size = batch_size as usize;
    let tree_size = tree_size as usize;
    let value_heads = value_heads as usize;

    for batch in 0..batch_size {
        let trie_batch = batch * tree_size;
        let batch_offset = batch * tree_size * value_heads;

        for row in 0..tree_size {
            let row_idx = row as u32;
            for head in 0..value_heads {
                let mut sum = 0.0f32;
                for col in 0..tree_size {
                    // col is an ancestor-or-self of row iff row falls in col's
                    // DFS subtree interval [trie_start, trie_end].
                    let node = unsafe { *trie.add(trie_batch + col) };
                    if row_idx < node.trie_start || row_idx > node.trie_end {
                        continue;
                    }
                    sum += unsafe { *log_decay.add(batch_offset + col * value_heads + head) };
                }
                unsafe { *prefix.add(batch_offset + row * value_heads + head) = sum };
            }
        }
    }
}
