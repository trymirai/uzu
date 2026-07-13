use std::sync::atomic::{AtomicBool, Ordering};

use crate::backends::common::SharedEvent;

const EVENT_WAIT_TIMEOUT_MS: u64 = 10;

pub(super) fn wait_until_event_or_cancelled<E: SharedEvent>(
    event: &E,
    value: u64,
    cancelled: &AtomicBool,
) -> bool {
    while event.signaled_value() < value {
        if cancelled.load(Ordering::Acquire) {
            return false;
        }
        event.wait_until_signaled_value_timeout_ms(value, EVENT_WAIT_TIMEOUT_MS);
    }
    !cancelled.load(Ordering::Acquire)
}

#[cfg(test)]
#[path = "../../../unit/encodable_block/per_layer_embedding/staging.rs"]
mod tests;
