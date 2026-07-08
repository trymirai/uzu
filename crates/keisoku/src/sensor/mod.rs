mod kind;
mod reading;

pub use kind::SensorKind;
pub use reading::Sensor;

use crate::sources::collect_sensors;

pub fn thermal_sensors() -> Box<[Sensor]> {
    sensors(SensorKind::Temperature)
}

fn sensors(kind: SensorKind) -> Box<[Sensor]> {
    collect_sensors(kind)
}
