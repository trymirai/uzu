use uzu_engine_macros::uzu_test;

use super::KVCacheState;

fn accept(
    state: &mut KVCacheState,
    indices: &[u32],
) -> Vec<(u32, u32)> {
    state.accept(indices).into_iter().map(|copy| (copy.source, copy.destination)).collect()
}

#[uzu_test]
fn ring_accepts_append_then_wrap() {
    let mut state = KVCacheState::ring(2);

    assert_eq!(accept(&mut state, &[1]), [(1, 0)]);
    assert_eq!(state.view().prefix_len(), 1);

    assert!(accept(&mut state, &[0]).is_empty());
    assert_eq!(state.view().prefix_len(), 2);

    assert_eq!(accept(&mut state, &[0]), [(2, 0)]);
    assert_eq!(state.view().ring_params().unwrap().ring_offset, 1);
    assert_eq!(state.required_prefix_len(99), 2);
}

#[uzu_test]
fn ring_accept_over_capacity_keeps_last() {
    let mut state = KVCacheState::ring(2);

    // Only the last `capacity` tokens survive; earlier copies are dead.
    assert_eq!(accept(&mut state, &[0, 1, 2]), [(2, 0)]);
    assert_eq!(state.view().prefix_len(), 2);
    assert_eq!(state.view().ring_params().unwrap().ring_offset, 1);

    // Same with a wrapped ring: destinations keep rotating.
    assert_eq!(accept(&mut state, &[0, 1, 2]), [(3, 0), (4, 1)]);
    assert_eq!(state.view().ring_params().unwrap().ring_offset, 0);
}
