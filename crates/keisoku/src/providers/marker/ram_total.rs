use super::{instant_set::InstantMetric, typelist::Metric};
use crate::{sources::Sources, units::Bytes};

pub struct RamTotal;

impl Metric for RamTotal {
    type Value = Bytes;
    const TYPE_BIT: u128 = 1 << 2;
}

impl InstantMetric for RamTotal {
    fn read(sources: &mut Sources) -> Bytes {
        Bytes(sources.system().total_memory())
    }
}
