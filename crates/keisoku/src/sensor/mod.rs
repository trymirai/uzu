mod kind;
mod reading;

pub use kind::SensorKind;
pub use reading::Sensor;

use crate::sys::hid::SensorReader;

pub fn thermal_sensors() -> Box<[Sensor]> {
    SensorReader::new(SensorKind::Temperature).map(|mut reader| reader.read()).unwrap_or_default()
}
