use tokenizers::{Tokenizer, models::wordlevel::WordLevel};
use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::gpu_types::trie::TrieNode as GpuTrieNode,
    encodable_block::sampling::PRng,
    engine::language_model::grammar::{Grammar, GrammarConfig},
    trie::{TrieNode, fill_bitmasks},
};

fn linearized_tree(
    rows: &[(u32, u32, u32)],
    padding: usize,
) -> (Vec<GpuTrieNode>, Vec<u32>, u32) {
    let count = rows.len() as u32;
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
    let mut token_ids: Vec<u32> = rows.iter().map(|&(token, _, _)| token).collect();
    for _ in 0..padding {
        nodes.push(GpuTrieNode {
            trie_start: u32::MAX,
            trie_end: 0,
            height: 0,
            parent: -1,
        });
        token_ids.push(0);
    }
    (nodes, token_ids, count)
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

fn verify_sprout(
    trie_root: &TrieNode,
    expected_seed: u64,
) {
    let flat_trie = trie_root.linearize();

    assert_eq!(flat_trie.len(), 1);
    assert_eq!(flat_trie.index(trie_root), Some(0));
    assert_eq!(flat_trie.index(&TrieNode::new(1, 0, 0.0)), None);
    assert_eq!(flat_trie.index(&TrieNode::new(0, 1, 0.0)), None);
    assert_eq!(flat_trie.index(&TrieNode::new(0, 0, 0.0)), None);
    assert_eq!(flat_trie.token_ids().collect::<Vec<u64>>(), vec![0]);
    assert_eq!(flat_trie.token_subtrie_ranges().map(|node| node.height as usize).collect::<Vec<_>>(), vec![0]);
    assert_eq!(flat_trie.token_seeds().collect::<Vec<u64>>(), vec![expected_seed]);
}

#[uzu_test]
fn test_trie_manual_sprout() {
    let trie_root = TrieNode::new(0, 0, 0.0);

    verify_sprout(&trie_root, 0);
}

fn verify_stick(
    trie_root: &TrieNode,
    rng: &PRng,
) {
    let flat_trie = trie_root.linearize();

    assert_eq!(flat_trie.len(), 10);
    let token_ids = flat_trie.token_ids().collect::<Vec<u64>>();
    let token_positions = flat_trie.token_subtrie_ranges().map(|node| node.height as usize).collect::<Vec<_>>();
    let token_seeds = flat_trie.token_seeds().collect::<Vec<u64>>();
    assert_eq!(token_ids.len(), 10);
    assert_eq!(token_positions.len(), 10);
    assert_eq!(token_seeds.len(), 10);

    let mut cur_node = trie_root;

    let position = flat_trie.index(cur_node).unwrap();
    assert_eq!(token_ids[position], 0);
    assert_eq!(token_positions[position], 0);
    assert_eq!(token_seeds[position], rng.derive(0));

    for i in 1..10 {
        cur_node = cur_node.get(i).unwrap();
        assert_eq!(cur_node.token(), i);
        assert_eq!(cur_node.seed, rng.derive(i));

        let position = flat_trie.index(cur_node).unwrap();
        assert_eq!(token_ids[position], cur_node.token());
        assert_eq!(token_positions[position], i as usize);
        assert_eq!(token_seeds[position], rng.derive(i));
    }
}

#[uzu_test]
fn test_trie_manual_stick() {
    let rng = PRng::new(0);

    let mut trie_stick = TrieNode::new(9, rng.derive(9), 0.0);
    for i in (1..9u64).rev() {
        let mut trie_parent = TrieNode::new(i, rng.derive(i), 0.0);
        trie_parent.add(trie_stick).unwrap();
        trie_stick = trie_parent;
    }
    let mut trie_root = TrieNode::new(0, rng.derive(0), 0.0);
    trie_root.add(trie_stick).unwrap();

    verify_stick(&trie_root, &rng);
}

fn verify_bush(
    trie_root: &TrieNode,
    rng: &PRng,
) {
    let flat_trie = trie_root.linearize();

    assert_eq!(flat_trie.len(), 4);
    let token_ids = flat_trie.token_ids().collect::<Vec<u64>>();
    let token_positions = flat_trie.token_subtrie_ranges().map(|node| node.height as usize).collect::<Vec<_>>();
    let token_seeds = flat_trie.token_seeds().collect::<Vec<u64>>();
    assert_eq!(token_ids.len(), 4);
    assert_eq!(token_positions.len(), 4);
    assert_eq!(token_seeds.len(), 4);

    let root_position = flat_trie.index(trie_root).unwrap();
    assert_eq!(token_ids[root_position], 0);
    assert_eq!(token_positions[root_position], 0);
    assert_eq!(token_seeds[root_position], rng.derive(0));

    for leaf_token in [1, 2, 3] {
        let leaf = trie_root.get(leaf_token).unwrap();
        assert_eq!(leaf.token(), leaf_token);
        assert_eq!(leaf.seed, rng.derive(1));

        let position = flat_trie.index(leaf).unwrap();
        assert_eq!(token_ids[position], leaf.token());
        assert_eq!(token_positions[position], 1);
        assert_eq!(token_seeds[position], rng.derive(1));
    }
}

#[uzu_test]
fn test_trie_manual_bush() {
    let rng = PRng::new(0);
    let mut trie_root = TrieNode::new(0, rng.derive(0), 0.0);

    assert!(trie_root.add(TrieNode::new(1, rng.derive(1), 0.0)).is_ok());
    assert!(trie_root.add(TrieNode::new(1, rng.derive(1), 0.0)).is_err());
    assert!(trie_root.add(TrieNode::new(1, 10, 0.0)).is_err());

    assert!(trie_root.add(TrieNode::new(2, rng.derive(1), 0.0)).is_ok());
    assert!(trie_root.add(TrieNode::new(3, rng.derive(1), 0.0)).is_ok());

    verify_bush(&trie_root, &rng);
}

fn verify_tree(
    trie_root: &TrieNode,
    rng: &PRng,
) {
    let flat_trie = trie_root.linearize();

    assert_eq!(flat_trie.len(), 7);
    let token_ids = flat_trie.token_ids().collect::<Vec<u64>>();
    let token_positions = flat_trie.token_subtrie_ranges().map(|node| node.height as usize).collect::<Vec<_>>();
    let token_seeds = flat_trie.token_seeds().collect::<Vec<u64>>();
    assert_eq!(token_ids.len(), 7);
    assert_eq!(token_positions.len(), 7);
    assert_eq!(token_seeds.len(), 7);

    let root_position = flat_trie.index(trie_root).unwrap();
    assert_eq!(token_ids[root_position], 0);
    assert_eq!(token_positions[root_position], 0);
    assert_eq!(token_seeds[root_position], rng.derive(0));

    for mid_token in [1, 2, 3] {
        let node = trie_root.get(mid_token).unwrap();
        let position = flat_trie.index(node).unwrap();
        assert_eq!(token_ids[position], mid_token);
        assert_eq!(token_positions[position], 1);
        assert_eq!(token_seeds[position], rng.derive(1));
    }

    for (mid_token, leaf_token) in [(2, 10), (3, 20), (3, 21)] {
        let leaf = trie_root.get(mid_token).unwrap().get(leaf_token).unwrap();
        let leaf_position = flat_trie.index(leaf).unwrap();
        assert_eq!(token_ids[leaf_position], leaf_token);
        assert_eq!(token_positions[leaf_position], 2);
        assert_eq!(token_seeds[leaf_position], rng.derive(2));
    }
}

#[uzu_test]
fn test_trie_manual_tree() {
    let rng = PRng::new(0);
    let mut trie_root = TrieNode::new(0, rng.derive(0), 0.0);

    assert!(trie_root.add(TrieNode::new(1, rng.derive(1), 0.0)).is_ok());
    assert!(trie_root.add(TrieNode::new(1, rng.derive(1), 0.0)).is_err());
    assert!(trie_root.add(TrieNode::new(1, 10, 0.0)).is_err());

    let mut mid_b = TrieNode::new(2, rng.derive(1), 0.0);
    assert!(mid_b.add(TrieNode::new(10, rng.derive(2), 0.0)).is_ok());

    let mut mid_c = TrieNode::new(3, rng.derive(1), 0.0);
    assert!(mid_c.add(TrieNode::new(20, rng.derive(2), 0.0)).is_ok());
    assert!(mid_c.add(TrieNode::new(21, rng.derive(2), 0.0)).is_ok());

    assert!(trie_root.add(mid_b).is_ok());
    assert!(trie_root.add(mid_c).is_ok());

    verify_tree(&trie_root, &rng)
}

fn verify_pruned(
    trie_root: &TrieNode,
    expected_tokens: &[u64],
) {
    assert_eq!(trie_root.node_count(), expected_tokens.len());
    assert_eq!(trie_root.linearize().token_ids().collect::<Vec<u64>>(), expected_tokens);
}

fn sample_tree() -> TrieNode {
    let mut root = TrieNode::new(0, 0, 0.0);
    let mut mid_a = TrieNode::new(1, 1, -0.1);
    mid_a.add(TrieNode::new(4, 2, -0.4)).unwrap();
    let mut mid_b = TrieNode::new(2, 1, -0.2);
    mid_b.add(TrieNode::new(5, 2, -2.8)).unwrap();
    root.add(mid_a).unwrap();
    root.add(mid_b).unwrap();
    root.add(TrieNode::new(3, 1, -0.3)).unwrap();
    root
}

#[uzu_test]
fn test_trie_prune_to_budget() {
    let mut trie = sample_tree();
    trie.prune_to_budget(4);
    verify_pruned(&trie, &[0, 1, 2, 3]);
    assert_eq!(trie.get(1).unwrap().logprob(), -0.1);
    assert_eq!(trie.get(2).unwrap().logprob(), -0.2);
    assert_eq!(trie.get(3).unwrap().logprob(), -0.3);

    let mut trie = sample_tree();
    trie.prune_to_budget(2);
    verify_pruned(&trie, &[0, 1]);

    let mut trie = sample_tree();
    trie.prune_to_budget(6);
    verify_pruned(&trie, &[0, 1, 4, 2, 5, 3]);

    let mut trie = sample_tree();
    trie.prune_to_budget(100);
    verify_pruned(&trie, &[0, 1, 4, 2, 5, 3]);
}

#[uzu_test]
fn fill_bitmasks_skips_dead_subtrees_and_restores_grammar() {
    let mut grammar = regex_grammar("ab", None);
    grammar.accept_token(0).unwrap();

    let mut expected_root_mask = [0];
    grammar.next_bitmask(&mut expected_root_mask);

    // (token, height, subtree end); token 0 is the committed root.
    let (nodes, token_ids, count) = linearized_tree(
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
    assert!(fill_bitmasks(&nodes, &token_ids, count, &mut masks, 9, &mut grammar));

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

    let (nodes, token_ids, count) = linearized_tree(&[(1, 0, 2), (0, 1, 1), (1, 2, 2)], 0);

    let mut masks = vec![0u32; nodes.len()];
    assert!(fill_bitmasks(&nodes, &token_ids, count, &mut masks, 9, &mut grammar));

    assert_eq!(masks, vec![1 << 7, u32::MAX, u32::MAX]);
    assert!(grammar.is_terminated());
}

#[uzu_test]
fn fill_bitmasks_leaves_masks_full_before_trigger() {
    let mut grammar = regex_grammar("ab", Some(vec![5, 6]));

    let (nodes, token_ids, count) = linearized_tree(&[(0, 0, 2), (3, 1, 2), (4, 2, 2)], 0);

    let mut masks = vec![0u32; nodes.len()];
    assert!(!fill_bitmasks(&nodes, &token_ids, count, &mut masks, 9, &mut grammar));

    assert_eq!(masks, vec![u32::MAX; 3]);
}

#[uzu_test]
fn test_trie_prune_to_budget_tie_keeps_parent() {
    let mut root = TrieNode::new(0, 0, 0.0);
    let mut child = TrieNode::new(1, 1, 0.0);
    child.add(TrieNode::new(2, 2, 0.0)).unwrap();
    root.add(child).unwrap();
    root.add(TrieNode::new(3, 1, 0.0)).unwrap();

    root.prune_to_budget(2);
    verify_pruned(&root, &[0, 1]);
}
