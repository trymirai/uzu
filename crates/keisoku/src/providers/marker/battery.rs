use super::{instant_set::InstantMetric, typelist::Metric};
use crate::{providers::data::BatteryMetrics, sources::Sources};

pub struct Battery;

impl Metric for Battery {
    type Value = Option<BatteryMetrics>;
    const TYPE_BIT: u128 = 1 << 7;
}

impl InstantMetric for Battery {
    fn read(sources: &mut Sources) -> Option<BatteryMetrics> {
        sources.battery()
    }
}
