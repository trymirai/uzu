use std::{marker::PhantomData, rc::Rc, time::Instant};

#[cfg(target_os = "macos")]
use crate::ioreport::RawEnergySample;

#[must_use]
pub struct EnergyWindow {
    pub(crate) started_at: Instant,
    #[cfg(target_os = "macos")]
    pub(crate) sample: RawEnergySample,
    _not_send: PhantomData<Rc<()>>,
}

impl EnergyWindow {
    #[cfg(target_os = "macos")]
    pub(crate) fn new(
        sample: RawEnergySample,
        started_at: Instant,
    ) -> Self {
        Self {
            started_at,
            sample,
            _not_send: PhantomData,
        }
    }
}
