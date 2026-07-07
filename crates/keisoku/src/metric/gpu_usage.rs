use std::time::Duration;

use obfstr::obfstr;

use super::{groups::IoReportGroups, measured::Measured};
use crate::{
    decode::{self, FrequencyTables, GroupId, RawChannel},
    metrics::GpuMetrics,
    sources::Sources,
    units::{Megahertz, Percent},
};

pub struct GpuUsage;

impl Measured for GpuUsage {
    type Value = GpuMetrics;
    type Ctx<'a> = FrequencyTables<'a>;
    type Acc = (u32, f32);
    const GROUPS: IoReportGroups = IoReportGroups::GPU_STATS;

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
        acc: &mut (u32, f32),
        channel: &RawChannel,
        ctx: &FrequencyTables<'_>,
    ) {
        if channel.group == GroupId::GpuStats
            && channel.subgroup == obfstr!("GPU Performance States")
            && channel.name == obfstr!("GPUPH")
            && ctx.gpu.len() > 1
        {
            *acc = decode::calculate_frequency(&channel.states, &ctx.gpu[1..]);
        }
    }

    fn finish(
        acc: (u32, f32),
        _elapsed: Duration,
        _ctx: &FrequencyTables<'_>,
    ) -> GpuMetrics {
        GpuMetrics {
            frequency: Megahertz(acc.0),
            usage: Percent(acc.1 * 100.0),
        }
    }
}
