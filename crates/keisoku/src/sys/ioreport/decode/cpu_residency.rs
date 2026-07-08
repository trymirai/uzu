use obfstr::obfstr;

use super::{ChannelFold, FrequencyTables, GroupId, RawChannel, calculate_frequency};
use crate::sys::ioreport::IoReportGroups;

#[derive(Default, Clone)]
pub(crate) struct CpuResidency {
    pub(crate) ecpu: Vec<(u32, f32)>,
    pub(crate) pcpu: Vec<(u32, f32)>,
}

impl ChannelFold for CpuResidency {
    const GROUPS: IoReportGroups = IoReportGroups::CPU_STATS;

    fn wants(channel: &RawChannel) -> bool {
        channel.group == GroupId::CpuStats
            && channel.subgroup == obfstr!("CPU Core Performance States")
            && (channel.name.starts_with(obfstr!("PCPU"))
                || channel.name.starts_with(obfstr!("ECPU"))
                || channel.name.starts_with(obfstr!("MCPU")))
    }

    fn fold(
        &mut self,
        channel: &RawChannel,
        frequencies: Option<&FrequencyTables<'_>>,
    ) {
        let Some(freq) = frequencies else { return };
        if channel.name.starts_with(obfstr!("PCPU")) {
            self.pcpu.push(calculate_frequency(&channel.states, freq.pcpu));
        } else if channel.name.starts_with(obfstr!("ECPU")) || channel.name.starts_with(obfstr!("MCPU")) {
            self.ecpu.push(calculate_frequency(&channel.states, freq.ecpu));
        }
    }
}
