//! Loading a speculator's confidence head: the head is kept only when it loads, fits the tree and was trained with
//! this speculator's weights file; otherwise the speculator runs without it instead of failing.

use std::path::{Path, PathBuf};

use uzu_engine_macros::uzu_test;

use super::{CONFIDENCE_HEAD_DIRECTORY, HeadTree, load_confidence_head};
use crate::{
    backends::cpu::Cpu,
    encodable_block::confidence_head::tests::{SMALL, Weights},
    tests::helpers::create_context,
};

/// sha256 of "abc" (FIPS 180-2), the bytes the tests write as the speculator's weights file.
const ABC_SHA256: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

/// The tree of the head tests' small head (SMALL): 10 rounds of 2 expansions (19 slots), 3 children, depth 4, a pool
/// of 64.
fn fixture_tree() -> HeadTree {
    HeadTree {
        rounds: 10,
        expand_per_round: 2,
        expand_width: 3,
        max_depth: 4,
        pool: 64,
    }
}

/// A speculator directory whose weights file holds "abc" and whose head is the head tests' small head with `edit`
/// applied to its config.
fn speculator(
    name: &str,
    edit: impl FnOnce(&mut serde_json::Value),
) -> PathBuf {
    let directory = std::env::temp_dir().join(format!("confidence_head_load_{}_{name}", std::process::id()));
    let head = directory.join(CONFIDENCE_HEAD_DIRECTORY);
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(directory.join("model.safetensors"), b"abc").unwrap();
    Weights::random(SMALL, 7).write(&head);
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(head.join("config.json")).unwrap()).unwrap();
    edit(&mut config);
    std::fs::write(head.join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();
    directory
}

fn trained_with(sha: &'static str) -> impl FnOnce(&mut serde_json::Value) {
    move |config| config["trained_with"]["speculator_sha256"] = serde_json::json!(sha)
}

fn loads(
    directory: &Path,
    tree: HeadTree,
) -> bool {
    let context = create_context::<Cpu>();
    let loaded = load_confidence_head::<Cpu>(directory, tree, &context).is_some();
    std::fs::remove_dir_all(directory).unwrap();
    loaded
}

/// Spec: a head trained with this speculator file and fitting the tree is kept.
#[uzu_test]
fn head_trained_with_this_speculator_is_kept() {
    assert!(loads(&speculator("same", trained_with(ABC_SHA256)), fixture_tree()));
}

/// Spec: a head trained with another speculator file is dropped.
#[uzu_test]
fn head_trained_with_another_speculator_is_dropped() {
    let other = "0000000000000000000000000000000000000000000000000000000000000000";
    assert!(!loads(&speculator("other", trained_with(other)), fixture_tree()));
}

/// Spec: a head that does not fit the tree (children per expansion, depth, pool, more slots than its attention takes:
/// 1 + 399 slots * 3 children > 512 tokens) is dropped, its speculator matching.
#[uzu_test]
fn head_that_does_not_fit_the_tree_is_dropped() {
    let mismatched = [
        HeadTree {
            rounds: 200,
            ..fixture_tree()
        },
        HeadTree {
            expand_width: 4,
            ..fixture_tree()
        },
        HeadTree {
            max_depth: 5,
            ..fixture_tree()
        },
        HeadTree {
            pool: 128,
            ..fixture_tree()
        },
    ];
    for (index, tree) in mismatched.into_iter().enumerate() {
        assert!(!loads(&speculator(&format!("tree{index}"), trained_with(ABC_SHA256)), tree), "{tree:?}");
    }
}

/// Spec: a head whose config does not load is dropped, not an error of the speculator.
#[uzu_test]
fn head_that_does_not_load_is_dropped() {
    let directory = speculator("format", |config| config["format"] = serde_json::json!(1));
    assert!(!loads(&directory, fixture_tree()));
}

/// Spec: a speculator without a head directory has no head.
#[uzu_test]
fn speculator_without_a_head_has_none() {
    let directory = speculator("none", trained_with(ABC_SHA256));
    std::fs::remove_dir_all(directory.join(CONFIDENCE_HEAD_DIRECTORY)).unwrap();
    assert!(!loads(&directory, fixture_tree()));
}
