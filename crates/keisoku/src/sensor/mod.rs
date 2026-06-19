//! IOHID per-sensor readings (temperature/voltage/current) and their kinds.

mod kind;
mod reading;

pub use kind::SensorKind;
pub use reading::Sensor;

pub fn thermal_sensors() -> Vec<Sensor> {
    crate::sensors(SensorKind::Temperature)
}

pub fn voltage_sensors() -> Vec<Sensor> {
    crate::sensors(SensorKind::Voltage)
}

pub fn current_sensors() -> Vec<Sensor> {
    crate::sensors(SensorKind::Current)
}
