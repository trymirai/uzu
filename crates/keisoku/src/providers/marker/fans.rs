use super::{instant_set::InstantMetric, typelist::Metric};
use crate::{providers::data::FanMetrics, sources::Sources};

pub struct Fans;

impl Metric for Fans {
    type Value = Option<FanMetrics>;
    const TYPE_BIT: u128 = 1 << 8;
}

impl InstantMetric for Fans {
    fn read(sources: &mut Sources) -> Option<FanMetrics> {
        sources.fans()
    }
}
