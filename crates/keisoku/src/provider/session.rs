use core::marker::PhantomData;
use std::time::Instant as Clock;

#[cfg(target_os = "macos")]
use crate::ioreport::RawEnergySample;
use crate::metric::Measured;

#[must_use]
pub struct Session<M: Measured> {
    #[cfg(target_os = "macos")]
    pub(super) begin: Option<RawEnergySample>,
    pub(super) begin_package_watts: Option<f32>,
    pub(super) started: Clock,
    pub(super) marker: PhantomData<M>,
}

#[cfg(target_os = "macos")]
unsafe impl<M: Measured> Send for Session<M> {}
