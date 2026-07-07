use std::time::Duration;

use obfstr::obfstr;

use super::{cpu_residency::CpuResidency, groups::IoReportGroups, measured::Measured};
use crate::{
    decode::{self, FrequencyTables, GroupId, RawChannel},
    metrics::CpuMetrics,
    sources::Sources,
    units::{Megahertz, Percent},
};

pub struct CpuUsage;

impl Measured for CpuUsage {
    type Value = CpuMetrics;
    type Ctx<'a> = FrequencyTables<'a>;
    type Acc = CpuResidency;
    const GROUPS: IoReportGroups = IoReportGroups::CPU_STATS;

    fn context(
        sources: &Sources,
        _package_watts_mean: Option<f32>,
    ) -> FrequencyTables<'_> {
        #[cfg(target_os = "macos")]
        {
            sources.frequencies()
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = sources;
            FrequencyTables::default()
        }
    }

    fn consume(
        acc: &mut CpuResidency,
        channel: &RawChannel,
        ctx: &FrequencyTables<'_>,
    ) {
        if channel.group == GroupId::CpuStats && channel.subgroup == obfstr!("CPU Core Performance States") {
            if channel.name.starts_with(obfstr!("PCPU")) {
                acc.pcpu.push(decode::calculate_frequency(&channel.states, ctx.pcpu));
            } else if channel.name.starts_with(obfstr!("ECPU")) || channel.name.starts_with(obfstr!("MCPU")) {
                acc.ecpu.push(decode::calculate_frequency(&channel.states, ctx.ecpu));
            }
        }
    }

    fn finish(
        acc: CpuResidency,
        _elapsed: Duration,
        ctx: &FrequencyTables<'_>,
    ) -> CpuMetrics {
        let ecpu_readings: Vec<(u32, f32)> = acc.ecpu.iter().copied().filter(|&(_, percent)| percent > 0.0).collect();
        let ecpu = decode::average_cluster_frequency(&ecpu_readings, ctx.ecpu);
        let pcpu = decode::average_cluster_frequency(&acc.pcpu, ctx.pcpu);
        let efficiency_cores = ctx.ecpu_cores as f32;
        let performance_cores = ctx.pcpu_cores as f32;
        let usage = decode::divide_or_zero(
            ecpu.1 * efficiency_cores + pcpu.1 * performance_cores,
            efficiency_cores + performance_cores,
        );
        CpuMetrics {
            usage: Percent(usage * 100.0),
            ecpu_frequency: Megahertz(ecpu.0),
            pcpu_frequency: Megahertz(pcpu.0),
        }
    }
}
