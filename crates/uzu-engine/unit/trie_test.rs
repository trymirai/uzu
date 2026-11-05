use tokenizers::{Tokenizer, models::wordlevel::WordLevel};
use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::gpu_types::trie::TrieNode as GpuTrieNode,
    engine::language_model::grammar::{Grammar, GrammarConfig},
    trie::fill_bitmasks,
};

fn linearized_tree(
    rows: &[(u32, u32, u32)],
    padding: usize,
) -> (Vec<GpuTrieNode>, Vec<u64>) {
    let mut stack: Vec<usize> = Vec::new();
    let mut nodes: Vec<GpuTrieNode> = rows
        .iter()
        .enumerate()
        .map(|(index, &(_, height, end))| {
            stack.truncate(height as usize);
            let parent = stack.last().map(|i| *i as i32).unwrap_or(-1);
            stack.push(index);
            GpuTrieNode {
                trie_start: index as u32,
                trie_end: end,
                height,
                parent,
            }
        })
        .collect();
    let mut token_ids: Vec<u64> = rows.iter().map(|&(token, _, _)| token as u64).collect();
    for _ in 0..padding {
        nodes.push(GpuTrieNode {
            trie_start: u32::MAX,
            trie_end: 0,
            height: 0,
            parent: -1,
        });
        token_ids.push(0);
    }
    (nodes, token_ids)
}

fn regex_grammar(
    pattern: &str,
    trigger: Option<Vec<u64>>,
) -> Grammar {
    let vocab = ["a", "b", "ab", "{", "}", "[", "]", "<eos>", "<unk>"]
        .into_iter()
        .enumerate()
        .map(|(id, token)| (token.to_string(), id as u32))
        .collect();
    let tokenizer = Tokenizer::new(WordLevel::builder().vocab(vocab).unk_token("<unk>".into()).build().unwrap());
    Grammar::new(&GrammarConfig::regex(pattern.into(), false), &tokenizer, trigger, Some(&[7])).unwrap()
}

#[uzu_test]
fn fill_bitmasks_skips_dead_subtrees_and_restores_grammar() {
    let mut grammar = regex_grammar("ab", None);
    grammar.accept_token(0).unwrap();

    let mut expected_root_mask = [0];
    grammar.next_bitmask(&mut expected_root_mask);

    // (token, height, subtree end); token 0 is the committed root.
    let (nodes, token_ids) = linearized_tree(
        &[
            (0, 0, 6), // 'a'
            (1, 1, 2), // 'b': completes "ab", terminates the grammar
            (0, 2, 2), // below a terminated row: unreachable
            (3, 1, 3), // '{': rejected
            (1, 1, 5), // 'b': completes "ab", terminates the grammar
            (0, 2, 5), // below a terminated row: unreachable
            (4, 1, 6), // '}': rejected
        ],
        1,
    );

    let mut masks = vec![0u32; nodes.len()];
    assert!(fill_bitmasks(&nodes, &token_ids, &mut masks, 9, &mut grammar));

    assert_eq!(masks, vec![0b010, 1 << 7, u32::MAX, u32::MAX, 1 << 7, u32::MAX, u32::MAX, u32::MAX]);

    let mut restored_mask = [0];
    grammar.next_bitmask(&mut restored_mask);
    assert_eq!(restored_mask, expected_root_mask);
    assert!(!grammar.is_terminated());
}

#[uzu_test]
fn fill_bitmasks_terminates_below_a_completed_grammar() {
    let mut grammar = regex_grammar("ab", None);
    grammar.accept_token(0).unwrap();
    grammar.accept_token(1).unwrap();
    assert!(grammar.is_terminated());

    let (nodes, token_ids) = linearized_tree(&[(1, 0, 2), (0, 1, 1), (1, 2, 2)], 0);

    let mut masks = vec![0u32; nodes.len()];
    assert!(fill_bitmasks(&nodes, &token_ids, &mut masks, 9, &mut grammar));

    assert_eq!(masks, vec![1 << 7, u32::MAX, u32::MAX]);
    assert!(grammar.is_terminated());
}

#[uzu_test]
fn fill_bitmasks_leaves_masks_full_before_trigger() {
    let mut grammar = regex_grammar("ab", Some(vec![5, 6]));

    let (nodes, token_ids) = linearized_tree(&[(0, 0, 2), (3, 1, 2), (4, 2, 2)], 0);

    let mut masks = vec![0u32; nodes.len()];
    assert!(!fill_bitmasks(&nodes, &token_ids, &mut masks, 9, &mut grammar));

    assert_eq!(masks, vec![u32::MAX; 3]);
}
