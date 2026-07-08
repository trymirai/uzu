use std::time::Instant as Clock;

use obfstr::obfstr;

use super::{frame::IntervalFrame, inputs::IntervalInputs};
use crate::{
    providers::marker::{CpuResidency, DramBandwidth, IoReportGroups},
    sources::Sources,
    sys::ioreport::{
        IoReport, RawEnergySample,
        decode::{self, EnergyTotals, FrequencyTables, GroupId, RawChannel},
    },
};

pub(crate) struct IntervalEngine {
    inputs: IntervalInputs,
    groups: IoReportGroups,
    ioreport: Option<IoReport>,
}

pub(crate) struct IntervalSession {
    begin: Option<RawEnergySample>,
    begin_package_watts: Option<f32>,
    started: Clock,
}

pub(crate) struct IntervalReading {
    begin: Option<RawEnergySample>,
    end: Option<RawEnergySample>,
    package_watts_mean: Option<f32>,
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

    pub(crate) fn begin(
        &self,
        sources: &Sources,
    ) -> IntervalSession {
        let begin = self.ioreport.as_ref().and_then(IoReport::snapshot);
        IntervalSession {
            begin,
            begin_package_watts: self.package_watts(sources),
            started: Clock::now(),
        }
    }

    pub(crate) fn end(
        &self,
        sources: &Sources,
        session: IntervalSession,
    ) -> IntervalReading {
        let end = self.ioreport.as_ref().and_then(IoReport::snapshot);
        let elapsed = session.started.elapsed();
        let package_watts_mean = self.package_watts_mean(sources, session.begin_package_watts);
        IntervalReading {
            begin: session.begin,
            end,
            package_watts_mean,
            elapsed,
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
        let mut gpu = self.inputs.contains(IntervalInputs::GPU_RESIDENCY).then(|| (0u32, 0.0f32));
        let mut ane = self.inputs.contains(IntervalInputs::ANE_ACTIVITY).then(|| 0.0f32);
        let mut bandwidth = self.inputs.contains(IntervalInputs::DRAM_BANDWIDTH).then(DramBandwidth::default);

        if let (Some(ioreport), Some(begin), Some(end)) =
            (self.ioreport.as_ref(), reading.begin.as_ref(), reading.end.as_ref())
        {
            ioreport.for_each_channel(
                begin,
                end,
                |channel| self.wants(channel),
                |channel| {
                    self.consume(
                        channel,
                        frequencies.as_ref(),
                        &mut energy,
                        &mut cpu,
                        &mut gpu,
                        &mut ane,
                        &mut bandwidth,
                    );
                },
            );
        }

        IntervalFrame {
            elapsed: reading.elapsed,
            energy,
            package_watts_mean: reading.package_watts_mean,
            cpu,
            gpu,
            ane,
            bandwidth,
            frequencies,
        }
    }

    fn wants(
        &self,
        channel: &RawChannel,
    ) -> bool {
        match channel.group {
            GroupId::EnergyModel if self.inputs.contains(IntervalInputs::ENERGY_RAILS) => true,
            GroupId::CpuStats
                if self.inputs.contains(IntervalInputs::CPU_RESIDENCY)
                    && channel.subgroup == obfstr!("CPU Core Performance States")
                    && (channel.name.starts_with(obfstr!("PCPU"))
                        || channel.name.starts_with(obfstr!("ECPU"))
                        || channel.name.starts_with(obfstr!("MCPU"))) =>
            {
                true
            },
            GroupId::GpuStats
                if self.inputs.contains(IntervalInputs::GPU_RESIDENCY)
                    && channel.subgroup == obfstr!("GPU Performance States")
                    && channel.name == obfstr!("GPUPH") =>
            {
                true
            },
            GroupId::Pmp
                if self.inputs.contains(IntervalInputs::ANE_ACTIVITY)
                    && channel.subgroup.contains(obfstr!("Floor"))
                    && (channel.name == obfstr!("ANE-AF-BW") || channel.name == obfstr!("ANE-DCS-BW")) =>
            {
                true
            },
            GroupId::Pmp
                if self.inputs.contains(IntervalInputs::DRAM_BANDWIDTH) && channel.subgroup == obfstr!("DRAM BW") =>
            {
                true
            },
            GroupId::AmcStats if self.inputs.contains(IntervalInputs::DRAM_BANDWIDTH) => {
                let aggregate = decode::strip_die_prefix(&channel.name);
                aggregate == obfstr!("DCS RD") || aggregate == obfstr!("DCS WR")
            },
            _ => false,
        }
    }

    fn consume(
        &self,
        channel: &RawChannel,
        frequencies: Option<&FrequencyTables<'_>>,
        energy: &mut Option<EnergyTotals>,
        cpu: &mut Option<CpuResidency>,
        gpu: &mut Option<(u32, f32)>,
        ane: &mut Option<f32>,
        bandwidth: &mut Option<DramBandwidth>,
    ) {
        if let Some(acc) = energy.as_mut()
            && channel.group == GroupId::EnergyModel
        {
            acc.accumulate(&channel.name, channel.integer_value, &channel.unit);
        }

        if let (Some(acc), Some(freq)) = (cpu.as_mut(), frequencies)
            && channel.group == GroupId::CpuStats
            && channel.subgroup == obfstr!("CPU Core Performance States")
        {
            if channel.name.starts_with(obfstr!("PCPU")) {
                acc.pcpu.push(decode::calculate_frequency(&channel.states, freq.pcpu));
            } else if channel.name.starts_with(obfstr!("ECPU")) || channel.name.starts_with(obfstr!("MCPU")) {
                acc.ecpu.push(decode::calculate_frequency(&channel.states, freq.ecpu));
            }
        }

        if let (Some(acc), Some(freq)) = (gpu.as_mut(), frequencies)
            && channel.group == GroupId::GpuStats
            && channel.subgroup == obfstr!("GPU Performance States")
            && channel.name == obfstr!("GPUPH")
            && freq.gpu.len() > 1
        {
            *acc = decode::calculate_frequency(&channel.states, &freq.gpu[1..]);
        }

        if let Some(acc) = ane.as_mut()
            && channel.group == GroupId::Pmp
            && channel.subgroup.contains(obfstr!("Floor"))
            && (channel.name == obfstr!("ANE-AF-BW") || channel.name == obfstr!("ANE-DCS-BW"))
        {
            *acc = acc.max(decode::residency_active_percent(&channel.states));
        }

        if let Some(acc) = bandwidth.as_mut() {
            if channel.group == GroupId::AmcStats {
                let bytes = channel.integer_value as f64;
                if bytes > 0.0 {
                    let aggregate = decode::strip_die_prefix(&channel.name);
                    if aggregate == obfstr!("DCS RD") {
                        acc.read_bytes += bytes;
                    } else if aggregate == obfstr!("DCS WR") {
                        acc.write_bytes += bytes;
                    }
                }
            } else if channel.group == GroupId::Pmp && channel.subgroup == obfstr!("DRAM BW") {
                let gbps = decode::residency_weighted_gbps(&channel.states);
                match decode::dram_flow(&channel.name) {
                    Some(true) => acc.read_histogram = acc.read_histogram.max(gbps),
                    Some(false) => acc.write_histogram = acc.write_histogram.max(gbps),
                    None => {},
                }
            }
        }
    }

    fn package_watts(
        &self,
        sources: &Sources,
    ) -> Option<f32> {
        self.inputs.needs_package_watts().then(|| sources.package_watts()).flatten().map(|watts| watts.value())
    }

    fn package_watts_mean(
        &self,
        sources: &Sources,
        begin_package_watts: Option<f32>,
    ) -> Option<f32> {
        match (begin_package_watts, self.package_watts(sources)) {
            (Some(first), Some(last)) => Some((first + last) / 2.0),
            _ => None,
        }
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
