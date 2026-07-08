use obfstr::obfstr;

use super::{ChannelFold, FrequencyTables, GroupId, RawChannel, Subgroup, residency_active_percent};
use crate::sys::ioreport::IoReportGroups;

#[derive(Default, Clone, Copy)]
pub(crate) struct AneActivity {
    pub(crate) active_percent: f32,
}

impl ChannelFold for AneActivity {
    const GROUPS: IoReportGroups = IoReportGroups::PMP;

    fn wants(channel: &RawChannel) -> bool {
        channel.group == GroupId::Pmp
            && Subgroup::classify(&channel.subgroup) == Subgroup::Floor
            && (channel.name == obfstr!("ANE-AF-BW") || channel.name == obfstr!("ANE-DCS-BW"))
    }

    fn fold(
        &mut self,
        channel: &RawChannel,
        _frequencies: Option<&FrequencyTables<'_>>,
    ) {
        self.active_percent = self.active_percent.max(residency_active_percent(&channel.states));
    }
}
