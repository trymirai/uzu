use itertools::Itertools;
use thiserror::Error;

use crate::{
    backends::common::gpu_types::trie::TrieNode as GpuTrieNode,
    data_type::DataType,
    engine::language_model::grammar::{Grammar, GrammarError},
};

#[derive(Debug, Error)]
pub enum TrieAcceptError {
    #[error("Grammar error: {0}")]
    Grammar(#[from] GrammarError),
}

pub fn fill_bitmasks(
    nodes: &[GpuTrieNode],
    tokens: &[u64],
    bitmasks: &mut [u32],
    vocab_size: usize,
    grammar: &mut Grammar,
) -> bool {
    let vocab_size_in_u32s = vocab_size.div_ceil(DataType::U32.size_in_bits());
    assert!(bitmasks.len() == nodes.len() * vocab_size_in_u32s);

    let mut any_non_full = false;
    // Height of the last row fed to the grammar; the grammar holds
    // `last_token_height + 1` tokens of the path currently walked.
    let mut last_token_height: isize = -1;
    // Net tokens pushed into the grammar during this call; rolled back at the
    // end so the grammar is left in its incoming state.
    let mut pushed = 0isize;
    // Rows before this index belong to a subtree the grammar cut off.
    let mut dead_until = 0usize;
    for ((token_index, &node), bitmask) in
        nodes.iter().enumerate().zip_eq(bitmasks.chunks_exact_mut(vocab_size_in_u32s))
    {
        if node.trie_start == u32::MAX || token_index < dead_until {
            // Padding row or dead subtree: unreachable, so unconstrained.
            bitmask.fill(u32::MAX);
            continue;
        }
        let height = node.height as isize;

        if token_index > 0 {
            if height <= last_token_height {
                let rolled_back = (last_token_height - height + 1) as usize;
                grammar.rollback(rolled_back);
                pushed -= rolled_back as isize;
            }
            if grammar.is_terminated() || grammar.accept_token(tokens[node.trie_start as usize]).is_err() {
                // Nothing below this row can ever be sampled: the subtree is dead.
                dead_until = node.trie_end as usize + 1;
                last_token_height = height - 1;
                bitmask.fill(u32::MAX);
                continue;
            }
            pushed += 1;
        }
        last_token_height = height;

        any_non_full |= grammar.next_bitmask(bitmask);
    }

    if pushed > 0 {
        grammar.rollback(pushed as usize);
    }

    any_non_full
}

pub fn accept(
    nodes: &[GpuTrieNode],
    output_tokens: &[u64],
    input_tokens: &[u32],
    mut grammar: Option<&mut Grammar>,
) -> Result<(Box<[u32]>, Box<[u64]>, Box<[u64]>), TrieAcceptError> {
    assert_eq!(nodes.len(), input_tokens.len());
    assert_eq!(nodes.len(), output_tokens.len());
    assert!(!nodes.is_empty(), "the trie needs at least a root");

    let mut accepted_indices = Vec::new();
    let mut accepted_input_token_ids = Vec::new();
    let mut accepted_output_token_ids = Vec::new();
    let mut current = 0usize;
    loop {
        let sampled = output_tokens[current];
        accepted_indices.push(current as u32);
        accepted_input_token_ids.push(input_tokens[current] as u64);
        accepted_output_token_ids.push(sampled);

        if let Some(grammar) = grammar.as_mut()
            && !grammar.is_terminated()
        {
            grammar.accept_token(sampled)?;
        }

        let end = nodes[current].trie_end as usize;
        let next = (current + 1..=end)
            .find(|&child| nodes[child].parent == current as i32 && input_tokens[child] as u64 == sampled);
        let Some(next) = next else {
            break;
        };
        current = next;
    }

    Ok((
        accepted_indices.into_boxed_slice(),
        accepted_input_token_ids.into_boxed_slice(),
        accepted_output_token_ids.into_boxed_slice(),
    ))
}

#[cfg(test)]
#[path = "../unit/trie_test.rs"]
mod tests;
