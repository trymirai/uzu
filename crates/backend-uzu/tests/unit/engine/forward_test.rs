use proc_macros::uzu_test;

use crate::{
    backends::common::gpu_types::trie::TrieNode as GpuTrieNode,
    engine::language_model::forward::{MockTree, chain_nodes},
};

fn mock(
    max_width: u32,
    balanced: bool,
) -> MockTree {
    MockTree {
        max_width,
        balanced,
    }
}

fn star_nodes(node_count: u32) -> Box<[GpuTrieNode]> {
    let mut nodes = Vec::with_capacity(node_count as usize);
    nodes.push(GpuTrieNode {
        trie_start: 0,
        trie_end: node_count - 1,
        height: 0,
    });
    for index in 1..node_count {
        nodes.push(GpuTrieNode {
            trie_start: index,
            trie_end: index,
            height: 1,
        });
    }
    nodes.into_boxed_slice()
}

#[uzu_test]
fn mock_tree_width_one_is_chain() {
    for node_count in [1, 2, 3, 7, 32] {
        let chain = chain_nodes(node_count);
        assert_eq!(mock(1, true).linearize(node_count).0, chain);
        assert_eq!(mock(1, false).linearize(node_count).0, chain);
    }
}

#[uzu_test]
fn mock_tree_unbalanced_width_above_budget_is_star() {
    for node_count in [2, 3, 5, 16] {
        let star = star_nodes(node_count);
        for max_width in node_count - 1..node_count + 3 {
            assert_eq!(mock(max_width, false).linearize(node_count).0, star);
        }
    }
}

#[uzu_test]
fn mock_tree_balanced_width_above_budget_is_uncapped() {
    for node_count in [2, 3, 5, 16] {
        let uncapped = mock(node_count, true).linearize(node_count).0;
        for max_width in node_count - 1..node_count + 3 {
            assert_eq!(mock(max_width, true).linearize(node_count).0, uncapped);
        }
    }
}

#[uzu_test]
fn mock_tree_growth_modes_differ() {
    assert_ne!(mock(2, true).linearize(6).0, mock(2, false).linearize(6).0);
}

#[uzu_test]
fn mock_tree_longest_path() {
    let (_, path) = mock(1, false).linearize(5);
    assert_eq!(&*path, &[0, 1, 2, 3, 4]);

    let (_, path) = mock(4, false).linearize(5);
    assert_eq!(&*path, &[0, 4]);

    let (nodes, path) = mock(2, false).linearize(6);
    assert_eq!(path.len(), 4);
    for (window, &index) in path.windows(2).zip(&path[1..]) {
        let parent = window[0];
        assert!(index > parent && index <= nodes[parent as usize].trie_end);
        assert_eq!(nodes[index as usize].height, nodes[parent as usize].height + 1);
    }
}
