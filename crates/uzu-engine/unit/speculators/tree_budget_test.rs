use std::collections::BTreeMap;

use uzu_engine_macros::uzu_test;

use super::{BudgetLatencies, LatenciesFile, TreeBudget};
use crate::trie::TrieNode;

/// A root whose children have the given path log-probabilities (no grandchildren).
fn star(logprobs: &[f32]) -> TrieNode {
    let mut root = TrieNode::new(0, 0, 0.0);
    for (token, &logprob) in (1u64..).zip(logprobs) {
        root.add(TrieNode::new(token, 0, logprob)).unwrap();
    }
    root
}

fn latencies(entries: &[(u32, f32)]) -> BudgetLatencies {
    BudgetLatencies::new(&entries.iter().copied().collect()).unwrap()
}

/// A budget is a node count or "dynamic", and is written back the same way; any other word is an error.
#[uzu_test]
fn tree_budget_reads_a_count_or_dynamic() {
    assert_eq!(serde_json::from_str::<TreeBudget>("16").unwrap(), TreeBudget::Fixed(16));
    assert_eq!(serde_json::from_str::<TreeBudget>(r#""dynamic""#).unwrap(), TreeBudget::Dynamic);
    assert!(serde_json::from_str::<TreeBudget>(r#""static""#).is_err());
    for budget in [TreeBudget::Fixed(32), TreeBudget::Dynamic] {
        let json = serde_json::to_string(&budget).unwrap();
        assert_eq!(serde_json::from_str::<TreeBudget>(&json).unwrap(), budget, "{json}");
    }
}

/// latencies.json names budgets by string keys, in any order; the budgets come out smallest first.
#[uzu_test]
fn latencies_read_budgets_from_string_keys() {
    let file: LatenciesFile = serde_json::from_str(r#"{"Apple M5 Pro": {"32": 75.4, "16": 68.9}}"#).unwrap();
    let step_ms = &file["Apple M5 Pro"];
    assert_eq!(step_ms.keys().copied().collect::<Vec<_>>(), [16, 32]);
    let latencies = BudgetLatencies::new(step_ms).unwrap();
    assert_eq!(latencies.largest(), 32);
    // Without a rate the smallest budget is taken, whatever the tree.
    assert_eq!(latencies.choose(&star(&[0.0, 0.0, 0.0]), None), 16);
}

/// No budgets, a budget below a root and one draft token, and a step time that is not a positive number are errors.
#[uzu_test]
fn budget_latencies_reject_bad_entries() {
    assert!(BudgetLatencies::new(&BTreeMap::new()).is_err());
    for entry in [(1, 10.0), (16, 0.0), (16, -1.0), (16, f32::NAN), (16, f32::INFINITY)] {
        assert!(BudgetLatencies::new(&[(32, 70.0), entry].into_iter().collect()).is_err(), "{entry:?}");
    }
}

/// The choice is the argmax over budgets of E_B - r / t_min * t_B. Children with probabilities 1, 1/2, 1/4 give
/// E_2 = 1, E_3 = 1.5, E_4 = 1.75; with t = 10, 11, 12 ms going from B to B + 1 adds 1/2 and then 1/4 token for
/// r / 10 tokens, so B = 4 below r = 2.5, B = 3 between 2.5 and 5, and B = 2 above 5 (from the definition, not the
/// code). Probabilities that are powers of two are exact in f32, so the boundaries are exact ties.
#[uzu_test]
fn choose_is_the_argmax_of_gain_net_of_time() {
    let trie = star(&[0.0, -std::f32::consts::LN_2, -2.0 * std::f32::consts::LN_2]);
    let latencies = latencies(&[(2, 10.0), (3, 11.0), (4, 12.0)]);
    for (rate, budget) in [(0.5, 4), (2.0, 4), (3.0, 3), (4.9, 3), (5.5, 2), (20.0, 2)] {
        assert_eq!(latencies.choose(&trie, Some(rate)), budget, "rate {rate}");
    }
}

/// Equal value goes to the smaller budget: children of probability 1 give E_2 = 1 and E_3 = 2; at t = 10 and 20 ms
/// and r = 1 both values are exactly 0.
#[uzu_test]
fn choose_breaks_ties_toward_the_smaller_budget() {
    let latencies = latencies(&[(2, 10.0), (3, 20.0)]);
    assert_eq!(latencies.choose(&star(&[0.0, 0.0]), Some(1.0)), 2);
    assert_eq!(latencies.choose(&star(&[0.0, 0.0]), Some(0.99)), 3);
}

/// A budget beyond the tree keeps the whole tree and gains nothing over it, so a slower larger budget never wins
/// over the one that already keeps every node.
#[uzu_test]
fn choose_does_not_pay_for_nodes_the_tree_lacks() {
    let latencies = latencies(&[(3, 10.0), (16, 11.0)]);
    assert_eq!(latencies.choose(&star(&[0.0, 0.0]), Some(0.01)), 3);
}
