use obfstr::obfstr;

use super::{
    ChannelFold, DramFlow, FrequencyTables, GroupId, RawChannel, Subgroup, dram_flow, residency_weighted_gbps,
    strip_die_prefix,
};
use crate::sys::ioreport::IoReportGroups;

#[derive(Default, Clone)]
pub(crate) struct DramBandwidth {
    pub(crate) read_bytes: f64,
    pub(crate) write_bytes: f64,
    pub(crate) read_histogram: f32,
    pub(crate) write_histogram: f32,
}

impl ChannelFold for DramBandwidth {
    const GROUPS: IoReportGroups = IoReportGroups::AMC_STATS.union(IoReportGroups::PMP);

    fn wants(channel: &RawChannel) -> bool {
        match channel.group {
            GroupId::AmcStats => {
                let aggregate = strip_die_prefix(&channel.name);
                aggregate == obfstr!("DCS RD") || aggregate == obfstr!("DCS WR")
            },
            GroupId::Pmp => Subgroup::classify(&channel.subgroup) == Subgroup::DramBandwidth,
            _ => false,
        }
    }

    fn fold(
        &mut self,
        channel: &RawChannel,
        _frequencies: Option<&FrequencyTables<'_>>,
    ) {
        match channel.group {
            GroupId::AmcStats => {
                let bytes = channel.integer_value as f64;
                if bytes > 0.0 {
                    let aggregate = strip_die_prefix(&channel.name);
                    if aggregate == obfstr!("DCS RD") {
                        self.read_bytes += bytes;
                    } else if aggregate == obfstr!("DCS WR") {
                        self.write_bytes += bytes;
                    }
                }
            },
            GroupId::Pmp => {
                let gbps = residency_weighted_gbps(&channel.states);
                match dram_flow(&channel.name) {
                    Some(DramFlow::Read) => self.read_histogram = self.read_histogram.max(gbps),
                    Some(DramFlow::Write) => self.write_histogram = self.write_histogram.max(gbps),
                    Some(DramFlow::Combined) | None => {},
                }
            },
            _ => {},
        }
    }
}
