use super::{instant_set::InstantMetric, typelist::Metric};
use crate::{sensor::Sensor, sources::Sources};

pub struct CurrentSensors;

impl Metric for CurrentSensors {
    type Value = Box<[Sensor]>;
    const TYPE_BIT: u128 = 1 << 13;
}

impl InstantMetric for CurrentSensors {
    fn read(sources: &mut Sources) -> Box<[Sensor]> {
        sources.current_sensors()
    }
}
