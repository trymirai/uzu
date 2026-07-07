use super::reading::Reading;
use crate::{sensor::Sensor, sources::Sources};

pub struct TemperatureSensors;

impl Reading for TemperatureSensors {
    type Value = Box<[Sensor]>;

    fn read(sources: &mut Sources) -> Box<[Sensor]> {
        sources.temperature_sensors()
    }
}
