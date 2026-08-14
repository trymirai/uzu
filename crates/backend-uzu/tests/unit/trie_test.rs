use proc_macros::uzu_test;

use crate::{encodable_block::sampling::PRng, trie::TrieNode};

fn verify_sprout(
    trie_root: &TrieNode,
    expected_seed: u64,
) {
    let flat_trie = trie_root.linearize();

    assert_eq!(flat_trie.len(), 1);
    assert_eq!(flat_trie.index(trie_root), Some(0));
    assert_eq!(flat_trie.index(&TrieNode::new(1, 0)), None);
    assert_eq!(flat_trie.index(&TrieNode::new(0, 1)), None);
    assert_eq!(flat_trie.index(&TrieNode::new(0, 0)), None);
    assert_eq!(flat_trie.token_ids().collect::<Vec<u64>>(), vec![0]);
    assert_eq!(flat_trie.token_subtrie_ranges().map(|node| node.height as usize).collect::<Vec<_>>(), vec![0]);
    assert_eq!(flat_trie.token_seeds().collect::<Vec<u64>>(), vec![expected_seed]);
}

#[uzu_test]
fn test_trie_manual_sprout() {
    let trie_root = TrieNode::new(0, 0);

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

    let mut trie_stick = TrieNode::new(9, rng.derive(9));
    for i in (1..9u64).rev() {
        let mut trie_parent = TrieNode::new(i, rng.derive(i));
        trie_parent.add(trie_stick).unwrap();
        trie_stick = trie_parent;
    }
    let mut trie_root = TrieNode::new(0, rng.derive(0));
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
    let mut trie_root = TrieNode::new(0, rng.derive(0));

    assert!(trie_root.add(TrieNode::new(1, rng.derive(1))).is_ok());
    assert!(trie_root.add(TrieNode::new(1, rng.derive(1))).is_err());
    assert!(trie_root.add(TrieNode::new(1, 10)).is_err());

    assert!(trie_root.add(TrieNode::new(2, rng.derive(1))).is_ok());
    assert!(trie_root.add(TrieNode::new(3, rng.derive(1))).is_ok());

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
    let mut trie_root = TrieNode::new(0, rng.derive(0));

    assert!(trie_root.add(TrieNode::new(1, rng.derive(1))).is_ok());
    assert!(trie_root.add(TrieNode::new(1, rng.derive(1))).is_err());
    assert!(trie_root.add(TrieNode::new(1, 10)).is_err());

    let mut mid_b = TrieNode::new(2, rng.derive(1));
    assert!(mid_b.add(TrieNode::new(10, rng.derive(2))).is_ok());

    let mut mid_c = TrieNode::new(3, rng.derive(1));
    assert!(mid_c.add(TrieNode::new(20, rng.derive(2))).is_ok());
    assert!(mid_c.add(TrieNode::new(21, rng.derive(2))).is_ok());

    assert!(trie_root.add(mid_b).is_ok());
    assert!(trie_root.add(mid_c).is_ok());

    verify_tree(&trie_root, &rng)
}

fn reference_accept(
    trie_root: &TrieNode,
    flat_trie: &crate::trie::FlatTrie,
    sampled_tokens: &[u64],
) -> Vec<(usize, u64, u64)> {
    let mut current = trie_root;
    let mut accepted = Vec::new();
    loop {
        let index = flat_trie.index(current).unwrap();
        let sampled = sampled_tokens[index];
        accepted.push((index, current.token(), sampled));
        let Some(next) = current.get(sampled) else {
            break;
        };
        current = next;
    }
    accepted
}

struct TestRng(u64);

impl TestRng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(
        &mut self,
        bound: u64,
    ) -> u64 {
        self.next() % bound
    }
}

fn grow_random_trie(
    node: &mut TrieNode,
    depth: usize,
    rng: &mut TestRng,
) {
    if depth == 0 {
        return;
    }
    let children = rng.below(4);
    for child in 0..children {
        let mut next = TrieNode::new(child, rng.next());
        grow_random_trie(&mut next, depth - 1, rng);
        node.add(next).unwrap();
    }
}

#[uzu_test]
fn test_trie_accept_matches_reference() {
    let mut rng = TestRng(0x9E3779B97F4A7C15);

    for case in 0..200 {
        let mut trie_root = TrieNode::new(0, rng.next());
        grow_random_trie(&mut trie_root, 1 + (case % 5), &mut rng);
        let flat_trie = trie_root.linearize();

        let sampled_tokens: Vec<u64> = (0..flat_trie.len()).map(|_| rng.below(5)).collect();

        let accepted = flat_trie
            .accept(
                &sampled_tokens,
                #[cfg(grammar)]
                None,
            )
            .unwrap();
        let expected = reference_accept(&trie_root, &flat_trie, &sampled_tokens);

        assert_eq!(accepted.as_ref(), expected.as_slice(), "case {case}");
    }
}

#[uzu_test]
fn test_trie_accept_full_chain() {
    let mut trie_root = TrieNode::new(0, 0);
    let mut node = &mut trie_root;
    for token in 1..32u64 {
        node.add(TrieNode::new(token, token)).unwrap();
        node = &mut node.next[0];
    }
    let flat_trie = trie_root.linearize();
    let sampled_tokens: Vec<u64> = (1..=flat_trie.len() as u64).collect();

    let accepted = flat_trie
        .accept(
            &sampled_tokens,
            #[cfg(grammar)]
            None,
        )
        .unwrap();
    let expected = reference_accept(&trie_root, &flat_trie, &sampled_tokens);

    assert_eq!(accepted.len(), 32, "the whole chain must be accepted");
    assert_eq!(accepted.as_ref(), expected.as_slice());
}
