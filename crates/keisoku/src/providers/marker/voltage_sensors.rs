use super::{instant_set::InstantMetric, typelist::Metric};
use crate::{sensor::Sensor, sources::Sources};

pub struct VoltageSensors;

impl Metric for VoltageSensors {
    type Value = Box<[Sensor]>;
    const TYPE_BIT: u128 = 1 << 12;
}

impl InstantMetric for VoltageSensors {
    fn read(sources: &mut Sources) -> Box<[Sensor]> {
        sources.voltage_sensors()
    }
}
