use core::marker::PhantomData;
use std::time::{Duration, Instant as Clock};

#[cfg(target_os = "macos")]
use crate::ioreport::{IoReport, RawEnergySample};
use crate::{
    decode::ChannelSample,
    metric::{IoReportGroups, Measured, Reading},
    sources::Sources,
};

pub struct Window<'a> {
    pub(crate) channels: &'a [ChannelSample],
    pub(crate) frequencies: Frequencies<'a>,
    pub(crate) elapsed: Duration,
    pub(crate) package_watts_mean: Option<f32>,
}

#[derive(Default, Clone, Copy)]
pub(crate) struct Frequencies<'a> {
    pub(crate) ecpu: &'a [u32],
    pub(crate) pcpu: &'a [u32],
    pub(crate) gpu: &'a [u32],
    pub(crate) ecpu_cores: u8,
    pub(crate) pcpu_cores: u8,
}

pub struct Static<M: Reading> {
    value: M::Value,
}

impl<M: Reading> Static<M> {
    pub fn new() -> Self {
        let mut sources = Sources::new();
        Self {
            value: M::read(&mut sources),
        }
    }

    pub fn get(&self) -> &M::Value {
        &self.value
    }

    pub fn into_inner(self) -> M::Value {
        self.value
    }
}

impl<M: Reading> Default for Static<M> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Instant<M: Reading> {
    sources: Sources,
    marker: PhantomData<M>,
}

impl<M: Reading> Instant<M> {
    pub fn new() -> Self {
        Self {
            sources: Sources::new(),
            marker: PhantomData,
        }
    }

    pub fn read(&mut self) -> M::Value {
        M::read(&mut self.sources)
    }
}

impl<M: Reading> Default for Instant<M> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Interval<M: Measured> {
    sources: Sources,
    #[cfg(target_os = "macos")]
    ioreport: Option<IoReport>,
    marker: PhantomData<M>,
}

#[cfg(target_os = "macos")]
unsafe impl<M: Measured> Send for Interval<M> {}

#[must_use]
pub struct Session<M: Measured> {
    #[cfg(target_os = "macos")]
    begin: Option<RawEnergySample>,
    begin_package_watts: Option<f32>,
    started: Clock,
    marker: PhantomData<M>,
}

#[cfg(target_os = "macos")]
unsafe impl<M: Measured> Send for Session<M> {}

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
        let package_watts_mean = self.package_mean(session.begin_package_watts);
        #[cfg(target_os = "macos")]
        let channels = self.decode_channels(&session);
        #[cfg(not(target_os = "macos"))]
        let channels: Box<[ChannelSample]> = Box::default();
        let frequencies = self.frequencies();
        let window = Window {
            channels: &channels,
            frequencies,
            elapsed,
            package_watts_mean,
        };
        M::extract(&window)
    }

    fn package_mean(
        &mut self,
        begin: Option<f32>,
    ) -> Option<f32> {
        match (begin, self.package_watts()) {
            (Some(first), Some(last)) => Some((first + last) / 2.0),
            _ => None,
        }
    }

    fn package_watts(&mut self) -> Option<f32> {
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

    #[cfg(target_os = "macos")]
    fn decode_channels(
        &self,
        session: &Session<M>,
    ) -> Box<[ChannelSample]> {
        let Some(begin) = session.begin.as_ref() else {
            return Box::default();
        };
        let Some(ioreport) = self.ioreport.as_ref() else {
            return Box::default();
        };
        let Some(end) = ioreport.snapshot() else {
            return Box::default();
        };
        ioreport.decode(begin, &end)
    }

    #[cfg(target_os = "macos")]
    fn frequencies(&mut self) -> Frequencies<'_> {
        if !M::GROUPS.intersects(IoReportGroups::CPU_STATS | IoReportGroups::GPU_STATS) {
            return Frequencies::default();
        }
        match self.sources.soc() {
            Some(soc) => Frequencies {
                ecpu: &soc.ecpu_frequencies,
                pcpu: &soc.pcpu_frequencies,
                gpu: &soc.gpu_frequencies,
                ecpu_cores: soc.ecpu_cores,
                pcpu_cores: soc.pcpu_cores,
            },
            None => Frequencies::default(),
        }
    }

    #[cfg(not(target_os = "macos"))]
    fn frequencies(&mut self) -> Frequencies<'_> {
        Frequencies::default()
    }
}

impl<M: Measured> Default for Interval<M> {
    fn default() -> Self {
        Self::new()
    }
}
