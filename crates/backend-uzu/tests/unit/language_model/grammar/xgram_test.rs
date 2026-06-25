use proc_macros::uzu_test;

use super::CompiledGrammarEngagementState;

#[uzu_test]
fn test_sequence_trigger_engages_only_after_full_sequence() {
    let trigger = [10u64, 11, 12];
    let mut state = CompiledGrammarEngagementState::from_config(&trigger, 256, 32);

    // Reasoning tokens, including the trailing sub-token of the tag appearing
    // mid-reasoning, must not engage the grammar.
    for token in [99, 12, 10, 11, 7] {
        state.accept_token(token);
        assert!(!state.is_engaged());
    }

    // The full sequence only engages once all sub-tokens arrive in order.
    state.accept_token(10);
    assert!(!state.is_engaged());
    state.accept_token(11);
    assert!(!state.is_engaged());
    state.accept_token(12);
    assert!(state.is_engaged());

    // Tokens accepted after engagement count toward the grammar distance.
    state.accept_token(42);
    state.accept_token(43);
    assert_eq!(state.rollback(0), 0);
    assert!(state.is_engaged());
}

#[uzu_test]
fn test_rollback_across_engagement_boundary() {
    let trigger = [10u64, 11, 12];
    let mut state = CompiledGrammarEngagementState::from_config(&trigger, 256, 32);

    for token in [10, 11, 12] {
        state.accept_token(token);
    }
    assert!(state.is_engaged());

    // Two JSON tokens accepted while engaged (distance == 2).
    state.accept_token(42);
    state.accept_token(43);

    // Rolling back 4 tokens crosses the boundary: only the 2 grammar tokens
    // are reported, and the state disengages while restoring the partial tag.
    let grammar_rollback = state.rollback(4);
    assert_eq!(grammar_rollback, 2);
    assert!(!state.is_engaged());

    // After disengaging, the partial-match prefix is preserved, so completing
    // the remaining sub-tokens of the tag re-engages exactly.
    state.accept_token(11);
    assert!(!state.is_engaged());
    state.accept_token(12);
    assert!(state.is_engaged());
}

#[uzu_test]
fn test_sequence_trigger_still_matches_after_long_reasoning_prefix() {
    let trigger = [10u64, 11, 12];
    let mut state = CompiledGrammarEngagementState::from_config(&trigger, 256, 32);

    for _ in 0..384 {
        state.accept_token(99);
        assert!(!state.is_engaged());
    }

    state.accept_token(10);
    state.accept_token(11);
    state.accept_token(12);
    assert!(state.is_engaged());
}
