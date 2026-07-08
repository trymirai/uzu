use super::{dram_bandwidth::DramBandwidth, interval_set::IntervalMetric, typelist::Metric};
use crate::{
    providers::metrics::BandwidthMetrics,
    sources::interval::{IntervalFrame, IntervalInputs},
    units::GigabytesPerSecond,
};

pub struct Bandwidth;

impl Metric for Bandwidth {
    type Value = BandwidthMetrics;
    const TYPE_BIT: u128 = 1 << 20;
}

impl IntervalMetric for Bandwidth {
    const INPUTS: IntervalInputs = IntervalInputs::DRAM_BANDWIDTH;

    fn finish(frame: &IntervalFrame<'_>) -> BandwidthMetrics {
        let acc = frame.bandwidth.as_ref().cloned().unwrap_or_default();
        project_bandwidth(acc, frame.elapsed)
    }
}

fn project_bandwidth(
    acc: DramBandwidth,
    elapsed: std::time::Duration,
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
