use super::{instant_set::InstantMetric, typelist::Metric};
use crate::{sensor::Sensor, sources::Sources};

pub struct TemperatureSensors;

impl Metric for TemperatureSensors {
    type Value = Box<[Sensor]>;
    const TYPE_BIT: u128 = 1 << 11;
}

impl InstantMetric for TemperatureSensors {
    fn read(sources: &mut Sources) -> Box<[Sensor]> {
        sources.temperature_sensors()
    }
}
