use std::time::Instant as Clock;

use super::{frame::IntervalFrame, inputs::IntervalInputs};
use crate::{
    sources::Sources,
    sys::ioreport::{
        IoReport, IoReportGroups, RawEnergySample,
        decode::{
            AneActivity, Channel, ChannelFold, CpuResidency, DramBandwidth, EnergyTotals, FrequencyTables,
            GpuResidency, RawChannel,
        },
    },
};

pub(crate) struct IntervalEngine {
    inputs: IntervalInputs,
    groups: IoReportGroups,
    ioreport: Option<IoReport>,
}

pub(crate) struct IntervalSession {
    begin: Option<RawEnergySample>,
    started: Clock,
}

pub(crate) struct IntervalReading {
    begin: Option<RawEnergySample>,
    end: Option<RawEnergySample>,
    elapsed: std::time::Duration,
}

impl IntervalEngine {
    pub(crate) fn new(inputs: IntervalInputs) -> Self {
        let groups = inputs.ioreport_groups();
        Self {
            inputs,
            groups,
            ioreport: (!groups.is_empty()).then(|| IoReport::for_groups(groups)).flatten(),
        }
    }

    pub(crate) fn is_available(&self) -> bool {
        if self.groups.is_empty() {
            return true;
        }
        self.ioreport.is_some()
    }

    pub(crate) fn prepare(
        &self,
        sources: &Sources,
    ) {
        if self.inputs.needs_frequencies() {
            #[cfg(target_os = "macos")]
            {
                let _ = sources.frequencies();
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = sources;
            }
        }
    }

    pub(crate) fn begin(&self) -> IntervalSession {
        IntervalSession {
            begin: self.ioreport.as_ref().and_then(IoReport::snapshot),
            started: Clock::now(),
        }
    }

    pub(crate) fn end(
        &self,
        session: IntervalSession,
    ) -> IntervalReading {
        IntervalReading {
            begin: session.begin,
            end: self.ioreport.as_ref().and_then(IoReport::snapshot),
            elapsed: session.started.elapsed(),
        }
    }

    pub(crate) fn frame<'a>(
        &self,
        sources: &'a Sources,
        reading: &IntervalReading,
    ) -> IntervalFrame<'a> {
        let frequencies = self.inputs.needs_frequencies().then(|| frequencies(sources));

        let mut energy = self.inputs.contains(IntervalInputs::ENERGY_RAILS).then(EnergyTotals::default);
        let mut cpu = self.inputs.contains(IntervalInputs::CPU_RESIDENCY).then(CpuResidency::default);
        let mut gpu = self.inputs.contains(IntervalInputs::GPU_RESIDENCY).then(GpuResidency::default);
        let mut ane = self.inputs.contains(IntervalInputs::ANE_ACTIVITY).then(AneActivity::default);
        let mut bandwidth = self.inputs.contains(IntervalInputs::DRAM_BANDWIDTH).then(DramBandwidth::default);

        if let (Some(ioreport), Some(begin), Some(end)) =
            (self.ioreport.as_ref(), reading.begin.as_ref(), reading.end.as_ref())
        {
            ioreport.for_each_channel(
                begin,
                end,
                |channel| self.inputs.wants(channel),
                |channel, raw| {
                    fold(&mut energy, channel, raw, frequencies.as_ref());
                    fold(&mut cpu, channel, raw, frequencies.as_ref());
                    fold(&mut gpu, channel, raw, frequencies.as_ref());
                    fold(&mut ane, channel, raw, frequencies.as_ref());
                    fold(&mut bandwidth, channel, raw, frequencies.as_ref());
                },
            );
        }

        IntervalFrame {
            elapsed: reading.elapsed,
            energy,
            cpu,
            gpu,
            ane,
            bandwidth,
            frequencies,
        }
    }
}

fn fold<T: ChannelFold>(
    accumulator: &mut Option<T>,
    channel: Channel,
    raw: &RawChannel,
    frequencies: Option<&FrequencyTables<'_>>,
) {
    if let Some(accumulator) = accumulator {
        accumulator.fold(channel, raw, frequencies);
    }
}

fn frequencies(sources: &Sources) -> FrequencyTables<'_> {
    #[cfg(target_os = "macos")]
    {
        sources.frequencies()
    }
    #[cfg(not(target_os = "macos"))]
    {
        FrequencyTables::default()
    }
}
