use super::{instant_set::InstantMetric, typelist::Metric};
use crate::{providers::data::ThermalPressure, sources::Sources};

pub struct Thermal;

impl Metric for Thermal {
    type Value = Option<ThermalPressure>;
    const TYPE_BIT: u128 = 1 << 10;
}

impl InstantMetric for Thermal {
    fn read(sources: &mut Sources) -> Option<ThermalPressure> {
        sources.thermal()
    }
}
