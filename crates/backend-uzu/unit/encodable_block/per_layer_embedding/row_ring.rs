use std::sync::atomic::Ordering;

use proc_macros::uzu_test;

use super::{RowRingState, SLOT_COMPLETING, SLOT_COUNT, SLOT_PENDING, SLOT_READY, reserve_slot};

#[uzu_test]
fn slots_are_linear_and_saturate_without_advancing() {
    let state = RowRingState::default();
    let mut generation = 0;
    for expected in 1..=SLOT_COUNT as u64 {
        assert_eq!(reserve_slot(&state, &mut generation).unwrap().0, expected);
    }

    state.slots[0].value.store(99, Ordering::Release);
    assert_eq!(reserve_slot(&state, &mut generation).unwrap_err().kind(), std::io::ErrorKind::WouldBlock);
    assert_eq!(generation, SLOT_COUNT as u64);
    assert_eq!(state.slots[0].value.load(Ordering::Acquire), 99);

    let slot = &state.slots[0];
    assert_eq!(slot.lifecycle.swap(SLOT_READY, Ordering::AcqRel), SLOT_PENDING);
    assert_eq!(
        slot.lifecycle.compare_exchange(SLOT_READY, SLOT_COMPLETING, Ordering::AcqRel, Ordering::Acquire),
        Ok(SLOT_READY)
    );
    assert_eq!(
        slot.lifecycle.compare_exchange(SLOT_READY, SLOT_COMPLETING, Ordering::AcqRel, Ordering::Acquire),
        Err(SLOT_COMPLETING)
    );
}
