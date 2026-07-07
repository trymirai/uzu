use std::time::Duration;

use obfstr::obfstr;

use super::{dram_bandwidth::DramBandwidth, groups::IoReportGroups, measured::Measured};
use crate::{
    decode::{self, GroupId, RawChannel},
    metrics::BandwidthMetrics,
    sources::Sources,
    units::GigabytesPerSecond,
};

pub struct Bandwidth;

impl Measured for Bandwidth {
    type Value = BandwidthMetrics;
    type Ctx<'a> = ();
    type Acc = DramBandwidth;
    const GROUPS: IoReportGroups = IoReportGroups::AMC_STATS.union(IoReportGroups::PMP);

    fn context(
        _sources: &Sources,
        _package_watts_mean: Option<f32>,
    ) {
    }

    fn consume(
        acc: &mut DramBandwidth,
        channel: &RawChannel,
        _ctx: &(),
    ) {
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

    fn finish(
        acc: DramBandwidth,
        elapsed: Duration,
        _ctx: &(),
    ) -> BandwidthMetrics {
        let seconds = elapsed.as_secs_f64().max(0.001);
        let to_gbps = |bytes: f64| (bytes / seconds / 1e9) as f32;
        let read = if acc.read_bytes > 0.0 {
            to_gbps(acc.read_bytes)
        } else {
            acc.read_histogram
        };
        let write = if acc.write_bytes > 0.0 {
            to_gbps(acc.write_bytes)
        } else {
            acc.write_histogram
        };
        BandwidthMetrics {
            dram_read: GigabytesPerSecond(read),
            dram_write: GigabytesPerSecond(write),
        }
    }
}
