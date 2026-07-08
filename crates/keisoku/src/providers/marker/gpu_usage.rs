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
        let acc = frame.gpu.unwrap_or_default();
        GpuMetrics {
            frequency: Megahertz(acc.frequency),
            usage: Percent(acc.usage * 100.0),
        }
    }
}
