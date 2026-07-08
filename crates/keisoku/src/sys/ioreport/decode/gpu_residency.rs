use obfstr::obfstr;

use super::{ChannelFold, FrequencyTables, GroupId, RawChannel, Subgroup, calculate_frequency};
use crate::sys::ioreport::IoReportGroups;

#[derive(Default, Clone, Copy)]
pub(crate) struct GpuResidency {
    pub(crate) frequency: u32,
    pub(crate) usage: f32,
}

impl ChannelFold for GpuResidency {
    const GROUPS: IoReportGroups = IoReportGroups::GPU_STATS;

    fn wants(channel: &RawChannel) -> bool {
        channel.group == GroupId::GpuStats
            && Subgroup::classify(&channel.subgroup) == Subgroup::GpuPerformanceStates
            && channel.name == obfstr!("GPUPH")
    }

    fn fold(
        &mut self,
        channel: &RawChannel,
        frequencies: Option<&FrequencyTables<'_>>,
    ) {
        let Some(freq) = frequencies else { return };
        if freq.gpu.len() <= 1 {
            return;
        }
        let (frequency, usage) = calculate_frequency(&channel.states, &freq.gpu[1..]);
        self.frequency = frequency;
        self.usage = usage;
    }
}
