use uzu_engine_macros::uzu_test;

use super::GrammarEngagementState;

// The muse-glimmer end-of-thinking tag:
// <|eom|> <|start|> assistant " to" "=user" <|message|>
const TRIGGER: [u64; 6] = [200007, 200022, 140680, 328, 76976, 200023];

fn triggered() -> GrammarEngagementState {
    GrammarEngagementState::Triggered {
        trigger_sequence: TRIGGER.to_vec(),
        match_history: Vec::new(),
    }
}

#[uzu_test]
fn always_is_engaged_and_rolls_back_everything() {
    let mut state = GrammarEngagementState::Always;
    assert!(state.is_engaged());
    state.accept_token(1);
    assert!(state.is_engaged());
    assert_eq!(state.rollback(3), 3);
}

#[uzu_test]
fn engages_only_after_the_full_trigger_sequence() {
    let mut state = triggered();
    // reasoning tokens, including a stray prefix of the trigger
    for token in [11, 200007, 200022, 42, 13] {
        state.accept_token(token);
        assert!(!state.is_engaged());
    }
    for token in TRIGGER {
        assert!(!state.is_engaged());
        state.accept_token(token);
    }
    assert!(state.is_engaged());
    // stays engaged for the rest of the stream
    state.accept_token(99);
    assert!(state.is_engaged());
}

#[uzu_test]
fn restarts_matching_on_a_mismatch_mid_sequence() {
    let mut state = triggered();
    for token in [TRIGGER[0], TRIGGER[1], TRIGGER[0], TRIGGER[1], TRIGGER[2], TRIGGER[3], TRIGGER[4], TRIGGER[5]] {
        state.accept_token(token);
    }
    // ..., eom, start, eom, start, assistant, to, =user, message: the second
    // eom restarts the match, which then runs to completion
    assert!(state.is_engaged());
}

#[uzu_test]
fn rollback_counts_only_tokens_accepted_while_engaged() {
    let mut state = triggered();
    state.accept_token(11);
    for token in TRIGGER {
        state.accept_token(token);
    }
    state.accept_token(90); // grammar token
    state.accept_token(91); // grammar token
    // 91, 90 were matcher-fed; <|message|> completed the trigger and was not
    assert_eq!(state.rollback(3), 2);
    assert!(!state.is_engaged());
    // the partial trigger state is restored: finishing it re-engages
    state.accept_token(TRIGGER[5]);
    assert!(state.is_engaged());
}

#[uzu_test]
fn rollback_past_the_start_is_safe() {
    let mut state = triggered();
    state.accept_token(11);
    assert_eq!(state.rollback(5), 0);
    assert!(!state.is_engaged());
}
