use core::marker::PhantomData;

use crate::{providers::metric::IntervalSet, sources::interval::IntervalSession};

#[must_use]
pub struct Session<M: IntervalSet> {
    pub(super) source: IntervalSession,
    pub(super) marker: PhantomData<M>,
}

#[cfg(target_os = "macos")]
unsafe impl<M: IntervalSet> Send for Session<M> {}
