//! Ports the acceptance cases from lalamo `tests/unit/speculator/test_proposal.py`
//! @ `2cb46e9` (expected values verbatim, minus batch dims and padding), plus
//! checks for invariants lalamo's tests leave implicit. The inactive-row and
//! gather_top_k cases are out of scope (batch machinery / top-k logits).

use proc_macros::uzu_test;

use crate::speculators::chain_acceptance::{AcceptedProposal, ChainProposal};

const NO_EOS: &[u64] = &[99999];

fn make_proposal() -> ChainProposal {
    ChainProposal {
        token_ids: Box::new([10, 11, 12, 13]),
        token_positions: Box::new([5, 6, 7, 8]),
    }
}

fn accept(
    sampled_token_ids: &[u64],
    eos_token_ids: &[u64],
) -> AcceptedProposal {
    make_proposal().accept(sampled_token_ids, 16, eos_token_ids)
}

#[uzu_test]
fn test_full_chain_accepted_with_bonus() {
    let accepted = accept(&[11, 12, 13, 14], NO_EOS);

    assert_eq!(accepted.token_ids, vec![11, 12, 13, 14]);
    assert_eq!(accepted.token_positions, vec![6, 7, 8, 9]);
    assert_eq!(accepted.num_accepted_nodes, 4);
    assert_eq!(accepted.accepted_node_indices(), 0..4);
}

#[uzu_test]
fn test_partial_chain_accepted() {
    let accepted = accept(&[11, 99, 13, 14], NO_EOS);

    assert_eq!(accepted.token_ids, vec![11, 99]);
    assert_eq!(accepted.token_positions, vec![6, 7]);
    assert_eq!(accepted.num_accepted_nodes, 2);
}

#[uzu_test]
fn test_all_drafts_rejected_emits_bonus_only() {
    let accepted = accept(&[99, 98, 97, 96], NO_EOS);

    assert_eq!(accepted.token_ids, vec![99]);
    assert_eq!(accepted.token_positions, vec![6]);
    assert_eq!(accepted.num_accepted_nodes, 1);
}

#[uzu_test]
fn test_eos_in_drafts_truncates_and_suppresses_bonus() {
    let accepted = accept(&[11, 12, 13, 14], &[12]);

    assert_eq!(accepted.token_ids, vec![11, 12]);
    assert!(accepted.has_eos(&[12]));
    // EOS truncates the committed output only; all four nodes stay verified
    // and keep advancing the draft state (common.py:131-134 vs 141-150).
    assert_eq!(accepted.num_accepted_nodes, 4);
}

#[uzu_test]
fn test_eos_at_first_accepted_draft() {
    let accepted = accept(&[11, 12, 13, 14], &[11]);

    assert_eq!(accepted.token_ids, vec![11]);
    assert!(accepted.has_eos(&[11]));
    assert_eq!(accepted.num_accepted_nodes, 4);
}

#[uzu_test]
fn test_eos_as_bonus_token() {
    let accepted = accept(&[11, 99, 13, 14], &[99]);

    assert_eq!(accepted.token_ids, vec![11, 99]);
    assert!(accepted.has_eos(&[99]));
    assert_eq!(accepted.num_accepted_nodes, 2);
}

#[uzu_test]
fn test_remaining_length_caps_emission() {
    let accepted = make_proposal().accept(&[11, 12, 13, 14], 2, NO_EOS);

    assert_eq!(accepted.token_ids, vec![11, 12]);
    // The output budget caps committed tokens only, not verified nodes.
    assert_eq!(accepted.num_accepted_nodes, 4);
}

#[uzu_test]
fn test_zero_remaining_length_commits_nothing() {
    let accepted = make_proposal().accept(&[11, 99, 13, 14], 0, NO_EOS);

    assert!(accepted.token_ids.is_empty());
    assert_eq!(accepted.num_accepted_nodes, 2);
    assert_eq!(accepted.last_token_id(), None);
    assert_eq!(accepted.last_token_index(), None);
}

#[uzu_test]
fn test_root_only_proposal_emits_bonus() {
    let proposal = ChainProposal {
        token_ids: Box::new([10]),
        token_positions: Box::new([5]),
    };
    let accepted = proposal.accept(&[42], 16, NO_EOS);

    assert_eq!(accepted.token_ids, vec![42]);
    assert_eq!(accepted.token_positions, vec![6]);
    assert_eq!(accepted.num_accepted_nodes, 1);
}

#[uzu_test]
fn test_last_token_id_and_index() {
    let accepted = accept(&[11, 99, 13, 14], NO_EOS);

    assert_eq!(accepted.last_token_id(), Some(99));
    assert_eq!(accepted.last_token_index(), Some(6));
}

#[uzu_test]
fn test_has_eos_negative() {
    let accepted = accept(&[11, 12, 13, 14], NO_EOS);

    assert!(!accepted.has_eos(&[12345]));
}
