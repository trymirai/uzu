use uzu_engine_macros::uzu_test;

use super::KVCacheState;

#[uzu_test]
fn ring_advance_appends_then_wraps() {
    let mut state = KVCacheState::ring(2);

    state.advance(1);
    assert_eq!(state.view().prefix_len(), 1);

    state.advance(1);
    assert_eq!(state.view().prefix_len(), 2);

    state.advance(1);
    assert_eq!(state.view().prefix_len(), 2);
    assert_eq!(state.view().ring_params().unwrap().ring_offset, 1);
    assert_eq!(state.required_prefix_len(99), 2);
}

#[uzu_test]
fn ring_advance_over_capacity_keeps_last() {
    let mut state = KVCacheState::ring(2);

    state.advance(3);
    assert_eq!(state.view().prefix_len(), 2);
    assert_eq!(state.view().ring_params().unwrap().ring_offset, 1);

    state.advance(3);
    assert_eq!(state.view().prefix_len(), 2);
    assert_eq!(state.view().ring_params().unwrap().ring_offset, 0);
}

#[uzu_test]
fn full_advance_extends_prefix() {
    let mut state = KVCacheState::full();

    state.advance(3);
    assert_eq!(state.view().prefix_len(), 3);

    state.advance(2);
    assert_eq!(state.view().prefix_len(), 5);
}
