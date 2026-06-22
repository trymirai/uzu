use serde::{Deserialize, Serialize};

#[cfg(target_vendor = "apple")]
use crate::sys;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
}

#[cfg(target_vendor = "apple")]
impl SensorKind {
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
