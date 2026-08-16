use proc_macros::uzu_test;

use crate::{encodable_block::sampling::PRng, trie::TrieNode};

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
fn test_trie_prune_to_budget_tie_keeps_parent() {
    let mut root = TrieNode::new(0, 0, 0.0);
    let mut child = TrieNode::new(1, 1, 0.0);
    child.add(TrieNode::new(2, 2, 0.0)).unwrap();
    root.add(child).unwrap();
    root.add(TrieNode::new(3, 1, 0.0)).unwrap();

    root.prune_to_budget(2);
    verify_pruned(&root, &[0, 1]);
}

#[uzu_test]
fn test_trie_prune_by_logprob_threshold() {
    let mut trie = sample_tree();
    trie.prune_by_logprob_threshold(-0.25);
    verify_pruned(&trie, &[0, 1, 2]);
}

#[uzu_test]
fn test_trie_prune_to_depth() {
    let mut trie = sample_tree();
    trie.prune_to_depth(1);
    verify_pruned(&trie, &[0, 1, 2, 3]);

    let mut trie = sample_tree();
    trie.prune_to_depth(0);
    verify_pruned(&trie, &[0]);
}
