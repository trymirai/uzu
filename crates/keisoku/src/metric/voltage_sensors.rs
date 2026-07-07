use super::reading::Reading;
use crate::{sensor::Sensor, sources::Sources};

pub struct VoltageSensors;

impl Reading for VoltageSensors {
    type Value = Box<[Sensor]>;

    fn read(sources: &mut Sources) -> Box<[Sensor]> {
        sources.voltage_sensors()
    }
}
