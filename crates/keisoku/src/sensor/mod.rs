mod kind;
mod reading;

use crate::sensors;
pub use kind::SensorKind;
pub use reading::Sensor;

pub fn thermal_sensors() -> Vec<Sensor> {
    sensors(SensorKind::Temperature)
}

pub fn voltage_sensors() -> Vec<Sensor> {
    sensors(SensorKind::Voltage)
}

pub fn current_sensors() -> Vec<Sensor> {
    sensors(SensorKind::Current)
}
