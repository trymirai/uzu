use core::marker::PhantomData;
use std::time::Instant as Clock;

use super::session::Session;
#[cfg(target_os = "macos")]
use crate::ioreport::IoReport;
use crate::{
    metric::{IoReportGroups, Measured},
    sources::Sources,
};

pub struct Interval<M: Measured> {
    sources: Sources,
    #[cfg(target_os = "macos")]
    ioreport: Option<IoReport>,
    marker: PhantomData<M>,
}

#[cfg(target_os = "macos")]
unsafe impl<M: Measured> Send for Interval<M> {}

impl<M: Measured> Interval<M> {
    pub fn new() -> Self {
        Self {
            sources: Sources::new(),
            #[cfg(target_os = "macos")]
            ioreport: (!M::GROUPS.is_empty()).then(|| IoReport::for_groups(M::GROUPS)).flatten(),
            marker: PhantomData,
        }
    }

    pub fn begin(&mut self) -> Session<M> {
        #[cfg(target_os = "macos")]
        let begin = self.ioreport.as_ref().and_then(IoReport::snapshot);
        Session {
            #[cfg(target_os = "macos")]
            begin,
            begin_package_watts: self.package_watts(),
            started: Clock::now(),
            marker: PhantomData,
        }
    }

    pub fn end(
        &mut self,
        session: Session<M>,
    ) -> M::Value {
        let elapsed = session.started.elapsed();
        let package = self.package_mean(&session);
        let ctx = M::context(&self.sources, package);
        #[cfg(target_os = "macos")]
        let acc = self.accumulate(&session, &ctx);
        #[cfg(not(target_os = "macos"))]
        let acc = M::Acc::default();
        M::finish(acc, elapsed, &ctx)
    }

    #[cfg(target_os = "macos")]
    fn accumulate(
        &self,
        session: &Session<M>,
        ctx: &M::Ctx<'_>,
    ) -> M::Acc {
        let mut acc = M::Acc::default();
        if let (Some(ioreport), Some(begin)) = (self.ioreport.as_ref(), session.begin.as_ref())
            && let Some(end) = ioreport.snapshot()
        {
            ioreport.for_each_channel(begin, &end, |channel| M::consume(&mut acc, channel, ctx));
        }
        acc
    }

    fn package_mean(
        &self,
        session: &Session<M>,
    ) -> Option<f32> {
        match (session.begin_package_watts, self.package_watts()) {
            (Some(first), Some(last)) => Some((first + last) / 2.0),
            _ => None,
        }
    }

    fn package_watts(&self) -> Option<f32> {
        if !M::GROUPS.contains(IoReportGroups::ENERGY_MODEL) {
            return None;
        }
        #[cfg(target_os = "macos")]
        {
            self.sources.smc().and_then(|smc| smc.package_watts()).map(|watts| watts.value())
        }
        #[cfg(not(target_os = "macos"))]
        {
            None
        }
    }
}

impl<M: Measured> Default for Interval<M> {
    fn default() -> Self {
        Self::new()
    }
}
