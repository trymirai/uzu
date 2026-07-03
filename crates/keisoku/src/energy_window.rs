use std::{marker::PhantomData, rc::Rc, time::Instant};

#[cfg(target_os = "macos")]
use crate::ioreport::RawEnergySample;

#[must_use]
pub struct EnergyWindow {
    pub(crate) started_at: Instant,
    #[cfg(target_os = "macos")]
    pub(crate) sample: RawEnergySample,
    #[cfg(target_os = "macos")]
    pub(crate) package_watts_start: Option<f32>,
    _not_send: PhantomData<Rc<()>>,
}

impl EnergyWindow {
    #[cfg(target_os = "macos")]
    pub(crate) fn new(
        sample: RawEnergySample,
        started_at: Instant,
        package_watts_start: Option<f32>,
    ) -> Self {
        Self {
            started_at,
            sample,
            package_watts_start,
            _not_send: PhantomData,
        }
    }
}
