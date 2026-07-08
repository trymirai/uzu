use super::{interval_set::IntervalMetric, typelist::Metric};
use crate::{
    providers::data::GpuMetrics,
    sources::interval::{IntervalFrame, IntervalInputs},
    units::{Megahertz, Percent},
};

pub struct GpuUsage;

impl Metric for GpuUsage {
    type Value = GpuMetrics;
    const TYPE_BIT: u128 = 1 << 18;
}

impl IntervalMetric for GpuUsage {
    const INPUTS: IntervalInputs = IntervalInputs::GPU_RESIDENCY.union(IntervalInputs::SOC_FREQUENCIES);

    fn finish(frame: &IntervalFrame<'_>) -> GpuMetrics {
        let acc = frame.gpu.unwrap_or((0, 0.0));
        GpuMetrics {
            frequency: Megahertz(acc.0),
            usage: Percent(acc.1 * 100.0),
        }
    }
}
