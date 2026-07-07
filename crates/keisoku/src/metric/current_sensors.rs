use super::reading::Reading;
use crate::{sensor::Sensor, sources::Sources};

pub struct CurrentSensors;

impl Reading for CurrentSensors {
    type Value = Box<[Sensor]>;

    fn read(sources: &mut Sources) -> Box<[Sensor]> {
        sources.current_sensors()
    }
}
