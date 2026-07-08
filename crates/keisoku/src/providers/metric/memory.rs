use super::{instant_set::InstantMetric, typelist::Metric};
use crate::{providers::metrics::MemoryMetrics, sources::Sources};

pub struct Memory;

impl Metric for Memory {
    type Value = Option<MemoryMetrics>;
    const TYPE_BIT: u128 = 1 << 6;
}

impl InstantMetric for Memory {
    fn read(sources: &mut Sources) -> Option<MemoryMetrics> {
        sources.memory()
    }
}
