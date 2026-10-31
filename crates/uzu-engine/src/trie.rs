use crate::{
    backends::common::gpu_types::trie::TrieNode as GpuTrieNode, data_type::DataType,
    engine::language_model::grammar::Grammar,
};

pub fn fill_bitmasks(
    nodes: &[GpuTrieNode],
    token_ids: &[u32],
    count: u32,
    bitmasks: &mut [u32],
    vocab_size: usize,
    grammar: &mut Grammar,
) -> bool {
    let words = vocab_size.div_ceil(DataType::U32.size_in_bits());
    let count = count as usize;
    assert_eq!(nodes.len(), token_ids.len());
    assert_eq!(bitmasks.len(), nodes.len() * words);
    assert!(count <= nodes.len());

    bitmasks[count * words..].fill(u32::MAX);

    let mut any_non_full = false;
    // Height of the path the walk has accepted so far; the matcher always
    // sits at the current row's parent whenever a row is rejected.
    let mut path_height = 0usize;
    let mut row = 0usize;
    while row < count {
        let height = nodes[row].height as usize;
        if row > 0 {
            assert!(height > 0, "non-root rows sit below the root");
            if path_height >= height {
                grammar.rollback(path_height + 1 - height);
                path_height = height - 1;
            }
            if grammar.accept_token(token_ids[row] as u64).is_err() {
                let end = nodes[row].trie_end as usize;
                bitmasks[row * words..(end + 1) * words].fill(u32::MAX);
                row = end + 1;
                continue;
            }
            path_height = height;
        }

        any_non_full |= grammar.next_bitmask(&mut bitmasks[row * words..(row + 1) * words]);

        if grammar.is_terminated() {
            let end = nodes[row].trie_end as usize;
            bitmasks[(row + 1) * words..(end + 1) * words].fill(u32::MAX);
            row = end + 1;
            continue;
        }
        row += 1;
    }

    if path_height > 0 {
        grammar.rollback(path_height);
    }

    any_non_full
}

#[cfg(test)]
#[path = "../unit/trie_test.rs"]
mod tests;
