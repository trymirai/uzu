use core::marker::PhantomData;

use super::session::Session;
use crate::{
    providers::marker::{IntervalSet, Sample},
    sources::{Sources, interval::IntervalEngine},
};

pub struct Interval<M: IntervalSet> {
    sources: Sources,
    engine: IntervalEngine,
    marker: PhantomData<M>,
}

#[cfg(target_os = "macos")]
unsafe impl<M: IntervalSet> Send for Interval<M> {}

impl<M: IntervalSet> Interval<M> {
    pub fn new() -> Self {
        let _ = M::TYPE_MASK;
        Self {
            sources: Sources::new(),
            engine: IntervalEngine::new(M::INPUTS),
            marker: PhantomData,
        }
    }

    pub fn is_available(&self) -> bool {
        self.engine.is_available()
    }

    pub fn try_new() -> Option<Self> {
        let interval = Self::new();
        interval.is_available().then_some(interval)
    }

    pub fn start(&mut self) -> Session<M> {
        self.engine.prepare(&self.sources);
        Session {
            source: self.engine.begin(&self.sources),
            marker: PhantomData,
        }
    }

    pub fn stop(
        &mut self,
        session: Session<M>,
    ) -> Sample<M> {
        let reading = self.engine.end(&self.sources, session.source);
        let frame = self.engine.frame(&self.sources, &reading);
        Sample::new(M::finish(&frame))
    }
}

impl<M: IntervalSet> Default for Interval<M> {
    fn default() -> Self {
        Self::new()
    }
}
