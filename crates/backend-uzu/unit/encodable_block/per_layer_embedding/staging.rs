use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

use proc_macros::uzu_test;

use super::{SharedEvent, wait_until_event_or_cancelled};

#[derive(Clone, Default)]
struct TestEvent {
    value: Arc<AtomicU64>,
    waits: Arc<AtomicU64>,
    cancel_on_wait: Option<Arc<AtomicBool>>,
}

impl SharedEvent for TestEvent {
    fn signaled_value(&self) -> u64 {
        self.value.load(Ordering::Acquire)
    }

    fn wait_until_signaled_value_timeout_ms(
        &self,
        value: u64,
        _timeout_ms: u64,
    ) -> bool {
        self.waits.fetch_add(1, Ordering::Relaxed);
        if let Some(cancelled) = &self.cancel_on_wait {
            cancelled.store(true, Ordering::Release);
            false
        } else {
            self.signal(value);
            true
        }
    }

    fn signal(
        &self,
        value: u64,
    ) {
        self.value.store(value, Ordering::Release);
    }
}

#[uzu_test]
fn waits_until_signaled() {
    let event = TestEvent::default();
    let cancelled = AtomicBool::new(false);

    assert!(wait_until_event_or_cancelled(&event, 1, &cancelled));
}

#[uzu_test]
fn observes_cancellation_after_wait() {
    let cancelled = Arc::new(AtomicBool::new(false));
    let event = TestEvent {
        cancel_on_wait: Some(Arc::clone(&cancelled)),
        ..TestEvent::default()
    };

    assert!(!wait_until_event_or_cancelled(&event, 1, &cancelled));
}
