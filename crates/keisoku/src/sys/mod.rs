#[cfg(target_os = "macos")]
mod battery;
mod constants;
#[cfg(target_os = "macos")]
mod memory;
#[cfg(not(target_os = "macos"))]
mod sysctl;
mod thermal;

#[cfg(target_os = "macos")]
pub(crate) mod ioreport;
#[cfg(target_os = "macos")]
pub(crate) mod registry;
#[cfg(target_os = "macos")]
pub(crate) mod smc;
#[cfg(target_os = "macos")]
pub(crate) mod soc;

pub(crate) mod hid;

#[cfg(target_os = "macos")]
pub(crate) use battery::read_battery;
pub(crate) use constants::{
    EVENT_TYPE_POWER, EVENT_TYPE_TEMPERATURE, HID_PAGE_APPLE_VENDOR, HID_PAGE_APPLE_VENDOR_POWER,
    HID_USAGE_POWER_CURRENT, HID_USAGE_POWER_VOLTAGE, HID_USAGE_TEMPERATURE_SENSOR, event_field_base,
};
#[cfg(target_os = "macos")]
pub(crate) use memory::read_memory;
#[cfg(not(target_os = "macos"))]
pub(crate) use sysctl::{perflevel_cores, sysctl_string};
pub(crate) use thermal::{ThermalPressureLevel, read_thermal_pressure};
