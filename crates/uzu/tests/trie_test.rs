use std::collections::{HashMap, HashSet};

use uzu::{
    language_model::rng::DerivableSeed,
    speculators::{empty_speculator::EmptySpeculator, speculator::Speculator},
    trie::{TrieCreationConfig, TrieNode},
};

struct StaticSpeculator {
    responses: HashMap<Vec<u64>, HashMap<u64, f32>>,
}

impl StaticSpeculator {
    fn new(responses: HashMap<Vec<u64>, HashMap<u64, f32>>) -> Self {
        Self {
            responses,
        }
    }
}

impl Speculator for StaticSpeculator {
    fn speculate(
        &self,
        prefix: &[u64],
    ) -> HashMap<u64, f32> {
        self.responses.get(prefix).cloned().unwrap_or_default()
    }
}

fn verify_sprout(
    trie_root: &TrieNode,
    expected_seed: u64,
) {
    let flat_trie = trie_root.linearize();

    assert_eq!(flat_trie.len(), 1);
    assert_eq!(flat_trie.index(&trie_root), Some(0));
    assert_eq!(flat_trie.index(&TrieNode::new(1, None, 0)), None);
    assert_eq!(flat_trie.index(&TrieNode::new(0, None, 1)), None);
    assert_eq!(flat_trie.index(&TrieNode::new(0, None, 0)), None);
    assert_eq!(flat_trie.token_ids().collect::<Vec<u64>>(), vec![0]);
    assert_eq!(flat_trie.token_positions().collect::<Vec<usize>>(), vec![0]);
    assert_eq!(
        flat_trie.token_seeds().collect::<Vec<u64>>(),
        vec![expected_seed]
    );
}

#[test]
fn test_trie_manual_sprout() {
    let trie_root = TrieNode::new(0, None, 0);

    verify_sprout(&trie_root, 0);
}

#[test]
fn test_trie_from_speculator_sprout() {
    let mut trie_rng = DerivableSeed::new(42);
    let mut ref_rng = DerivableSeed::new(42);

    let speculator = EmptySpeculator;

    let trie_root = TrieNode::from_speculator(
        &[0],
        &mut trie_rng,
        None,
        &speculator,
        &TrieCreationConfig {
            width: 5,
        },
        10,
    );

    verify_sprout(&trie_root, ref_rng.next());
}

fn verify_stick(
    trie_root: &TrieNode,
    mut expected_seeds: HashSet<u64>,
) {
    let flat_trie = trie_root.linearize();

    assert_eq!(flat_trie.len(), 10);
    let token_ids = flat_trie.token_ids().collect::<Vec<u64>>();
    let token_positions = flat_trie.token_positions().collect::<Vec<usize>>();
    let token_seeds = flat_trie.token_seeds().collect::<Vec<u64>>();
    assert_eq!(token_ids.len(), 10);
    assert_eq!(token_positions.len(), 10);
    assert_eq!(token_seeds.len(), 10);

    let mut cur_node = trie_root;

    let position = flat_trie.index(cur_node).unwrap();
    assert_eq!(token_ids[position], 0);
    assert_eq!(token_positions[position], 0);
    assert!(expected_seeds.remove(&token_seeds[position]));

    for i in 1..10 {
        cur_node = cur_node.get(i as u64).unwrap();
        assert_eq!(cur_node.token(), i as u64);
        assert!(expected_seeds.remove(&cur_node.seed()));

        let position = flat_trie.index(cur_node).unwrap();
        assert_eq!(token_ids[position], cur_node.token());
        assert_eq!(token_positions[position], i as usize);
        assert_eq!(token_seeds[position], cur_node.seed());
    }

    assert!(expected_seeds.is_empty());
}

#[test]
fn test_trie_manual_stick() {
    let mut trie_root = TrieNode::new(0, None, 0);

    let mut trie_leaf = &mut trie_root;
    for (token, seed) in std::iter::zip(1..10, 1..10) {
        trie_leaf.add(TrieNode::new(token, None, seed)).unwrap();
        trie_leaf = trie_leaf.get_mut(token).unwrap();
    }

    verify_stick(&trie_root, (0..10).collect());
}

#[test]
fn test_trie_from_speculator_stick() {
    let mut trie_rng = DerivableSeed::new(42);
    let mut ref_rng = DerivableSeed::new(42);

    let mut hs = HashMap::new();
    for i in 1..10 {
        hs.insert(
            (0..i).collect(),
            vec![(i, 0.988), (888, 0.001), (999, 0.001)].into_iter().collect(),
        );
    }
    let speculator = StaticSpeculator::new(hs);

    let trie_root = TrieNode::from_speculator(
        &[0],
        &mut trie_rng,
        None,
        &speculator,
        &TrieCreationConfig {
            width: 1,
        },
        10,
    );

    verify_stick(
        &trie_root,
        std::iter::from_fn(|| Some(ref_rng.next())).take(10).collect(),
    );
}

fn verify_bush(
    trie_root: &TrieNode,
    mut expected_seeds: HashSet<u64>,
) {
    let flat_trie = trie_root.linearize();

    assert_eq!(flat_trie.len(), 4);
    let token_ids = flat_trie.token_ids().collect::<Vec<u64>>();
    let token_positions = flat_trie.token_positions().collect::<Vec<usize>>();
    let token_seeds = flat_trie.token_seeds().collect::<Vec<u64>>();
    assert_eq!(token_ids.len(), 4);
    assert_eq!(token_positions.len(), 4);
    assert_eq!(token_seeds.len(), 4);

    let root_position = flat_trie.index(&trie_root).unwrap();
    assert_eq!(token_ids[root_position], 0);
    assert_eq!(token_positions[root_position], 0);
    assert!(expected_seeds.remove(&token_seeds[root_position]));

    for leaf_token in [1, 2, 3] {
        let leaf = trie_root.get(leaf_token).unwrap();
        assert_eq!(leaf.token(), leaf_token);
        assert!(expected_seeds.remove(&leaf.seed()));

        let position = flat_trie.index(&leaf).unwrap();
        assert_eq!(token_ids[position], leaf.token());
        assert_eq!(token_positions[position], 1);
        assert_eq!(token_seeds[position], leaf.seed());
    }

    assert!(expected_seeds.is_empty());
}

#[test]
fn test_trie_manual_bush() {
    let mut trie_root = TrieNode::new(0, None, 0);

    assert!(trie_root.add(TrieNode::new(1, None, 1)).is_ok());
    assert!(trie_root.add(TrieNode::new(1, None, 1)).is_err());
    assert!(trie_root.add(TrieNode::new(1, None, 10)).is_err());

    assert!(trie_root.add(TrieNode::new(2, None, 2)).is_ok());
    assert!(trie_root.add(TrieNode::new(3, None, 3)).is_ok());

    verify_bush(&trie_root, (0..4).into_iter().collect());
}

#[test]
fn test_trie_from_speculator_bush() {
    let mut trie_rng = DerivableSeed::new(42);
    let mut ref_rng = DerivableSeed::new(42);

    let mut hs = HashMap::new();
    hs.insert(
        vec![0],
        vec![(1, 0.33), (2, 0.33), (3, 0.33), (888, 0.005), (999, 0.005)]
            .into_iter()
            .collect(),
    );
    let speculator = StaticSpeculator::new(hs);

    let trie_root = TrieNode::from_speculator(
        &[0],
        &mut trie_rng,
        None,
        &speculator,
        &TrieCreationConfig {
            width: 3,
        },
        10,
    );

    verify_bush(
        &trie_root,
        std::iter::from_fn(|| Some(ref_rng.next())).take(4).collect(),
    );
}

fn verify_tree(trie_root: &TrieNode) {
    let flat_trie = trie_root.linearize();

    assert_eq!(flat_trie.len(), 7);
    let token_ids = flat_trie.token_ids().collect::<Vec<u64>>();
    let token_positions = flat_trie.token_positions().collect::<Vec<usize>>();
    let token_seeds = flat_trie.token_seeds().collect::<Vec<u64>>();
    assert_eq!(token_ids.len(), 7);
    assert_eq!(token_positions.len(), 7);
    assert_eq!(token_seeds.len(), 7);

    let root_position = flat_trie.index(&trie_root).unwrap();
    assert_eq!(token_ids[root_position], 0);
    assert_eq!(token_positions[root_position], 0);
    assert_eq!(token_seeds[root_position], 0);

    for (mid_token, mid_seed) in [(1, 1), (2, 1), (3, 2)] {
        let leaf = trie_root.get(mid_token).unwrap();
        let position = flat_trie.index(&leaf).unwrap();
        assert_eq!(token_ids[position], mid_token);
        assert_eq!(token_positions[position], 1);
        assert_eq!(token_seeds[position], mid_seed);
    }

    for (mid_token, leaf_token, leaf_seed) in
        [(2, 10, 15), (3, 20, 25), (3, 21, 26)]
    {
        let leaf = trie_root.get(mid_token).unwrap().get(leaf_token).unwrap();
        let leaf_position = flat_trie.index(leaf).unwrap();
        assert_eq!(token_ids[leaf_position], leaf_token);
        assert_eq!(token_positions[leaf_position], 2);
        assert_eq!(token_seeds[leaf_position], leaf_seed);
    }
}

#[test]
fn test_trie_manual_tree() {
    let mut trie_root = TrieNode::new(0, None, 0);

    assert!(trie_root.add(TrieNode::new(1, None, 1)).is_ok());
    assert!(trie_root.add(TrieNode::new(1, None, 1)).is_err());
    assert!(trie_root.add(TrieNode::new(1, None, 10)).is_err());

    assert!(trie_root.add(TrieNode::new(2, None, 1)).is_ok());
    assert!(trie_root.add(TrieNode::new(3, None, 2)).is_ok());

    let mid_b = trie_root.get_mut(2).unwrap();
    assert!(mid_b.add(TrieNode::new(10, None, 15)).is_ok());

    let mid_c = trie_root.get_mut(3).unwrap();
    assert!(mid_c.add(TrieNode::new(20, None, 25)).is_ok());
    assert!(mid_c.add(TrieNode::new(21, None, 26)).is_ok());

    verify_tree(&trie_root)
}
