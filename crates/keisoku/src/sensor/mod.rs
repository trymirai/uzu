mod kind;
mod reading;

use crate::sensors;
pub use kind::SensorKind;
pub use reading::Sensor;

pub fn thermal_sensors() -> Box<[Sensor]> {
    sensors(SensorKind::Temperature)
}

pub fn voltage_sensors() -> Box<[Sensor]> {
    sensors(SensorKind::Voltage)
}

pub fn current_sensors() -> Box<[Sensor]> {
    sensors(SensorKind::Current)
}
