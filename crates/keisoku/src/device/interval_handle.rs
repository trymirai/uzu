use core::marker::PhantomData;
use std::time::Instant;

use crate::{
    marker::{IntervalSet, Sample},
    sources::interval::{IntervalEngine, IntervalSession},
};

/// Handle for measuring selected IOReport channels over a time window.
#[must_use]
pub struct IntervalHandle<M: IntervalSet> {
    engine: IntervalEngine,
    session: Option<IntervalSession>,
    started: Option<Instant>,
    marker: PhantomData<M>,
}

// SAFETY: `IntervalHandle` wraps CoreFoundation IOReport objects that are not `Sync`.
// Callers must use a handle only from the thread that created it (same contract as the
// previous `Interval`/`Session` types). `Send` is asserted so `PowerMeter` can live in
// `RefCell` behind `Send` trait objects such as nagare's `PowerRecorder`.
unsafe impl<M: IntervalSet> Send for IntervalHandle<M> {}

impl<M: IntervalSet> IntervalHandle<M> {
    pub(super) fn new() -> Self {
        let _ = M::TYPE_MASK;
        Self {
            engine: IntervalEngine::new(M::GROUPS),
            session: None,
            started: None,
            marker: PhantomData,
        }
    }

    /// Begins an IOReport snapshot. Calling again without [`stop`](Self::stop) restarts the window.
    pub fn start(&mut self) {
        debug_assert!(self.session.is_none(), "start without stop");
        drop(self.session.take());
        self.started = Some(Instant::now());
        self.session = Some(self.engine.begin());
    }

    /// Ends the measurement and returns channel deltas, or `None` if [`start`](Self::start) was not called.
    pub fn stop(&mut self) -> Option<Sample<M>> {
        let session = self.session.take()?;
        let mut values = M::default_values();
        self.engine.fold_end::<M>(session, &mut values);
        Some(Sample::new(values))
    }

    pub(crate) fn elapsed(&self) -> std::time::Duration {
        self.started.map(|started| started.elapsed()).unwrap_or_default()
    }
}
