use crate::{component::Component, sys};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SensorKind {
    Temperature,
    Voltage,
    Current,
}

impl SensorKind {
    pub fn unit(self) -> &'static str {
        match self {
            SensorKind::Temperature => "°C",
            SensorKind::Voltage => "V",
            SensorKind::Current => "A",
        }
    }

    pub(crate) fn matching(self) -> (i32, i32) {
        match self {
            SensorKind::Temperature => (sys::HID_PAGE_APPLE_VENDOR, sys::HID_USAGE_TEMPERATURE_SENSOR),
            SensorKind::Voltage => (sys::HID_PAGE_APPLE_VENDOR_POWER, sys::HID_USAGE_POWER_VOLTAGE),
            SensorKind::Current => (sys::HID_PAGE_APPLE_VENDOR_POWER, sys::HID_USAGE_POWER_CURRENT),
        }
    }

    pub(crate) fn event_type(self) -> i64 {
        match self {
            SensorKind::Temperature => sys::EVENT_TYPE_TEMPERATURE,
            SensorKind::Voltage | SensorKind::Current => sys::EVENT_TYPE_POWER,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Sensor {
    pub name: String,
    pub value: f64,
    pub kind: SensorKind,
    pub component: Component,
    pub manufacturer: Option<String>,
    pub category: Option<String>,
    pub location_id: Option<i64>,
    pub registry_id: u64,
}

pub fn thermal_sensors() -> Vec<Sensor> {
    crate::sensors(SensorKind::Temperature)
}

pub fn voltage_sensors() -> Vec<Sensor> {
    crate::sensors(SensorKind::Voltage)
}

pub fn current_sensors() -> Vec<Sensor> {
    crate::sensors(SensorKind::Current)
}
