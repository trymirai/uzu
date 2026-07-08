use super::{instant_set::InstantMetric, typelist::Metric};
use crate::{sources::Sources, units::Watts};

pub struct RailPower;

impl Metric for RailPower {
    type Value = Option<Watts>;
    const TYPE_BIT: u128 = 1 << 14;
}

impl InstantMetric for RailPower {
    fn read(sources: &mut Sources) -> Option<Watts> {
        sources.rail_power()
    }
}
