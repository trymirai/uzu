// Apple platforms only (macOS and iOS). Elsewhere the crate is empty, so workspace-wide
// builds (`cargo build`, `cargo clippy --workspace`) still work on Windows and Linux.
#![cfg(target_vendor = "apple")]

mod device;
#[cfg(target_os = "macos")]
mod marker;
mod metrics;
mod sources;
mod sys;

mod component;
mod error;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod power_meter;
mod sensor;
mod units;

pub use component::{Component, classify};
pub use device::Device;
pub use error::KeisokuError;
#[cfg(target_os = "macos")]
pub use marker::{
    Ane, AneBandwidth, Cons, Cpu, DramBytes, DramHistogram, DramRead, DramWrite, EnergyRail, Gpu, Nil, Ram, Sample,
};
pub use metrics::{BatteryMetrics, Fan, FanMetrics, MemoryMetrics, ThermalPressure};
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use power_meter::{PowerMeter, PowerReading};
pub use sensor::{Sensor, SensorKind, thermal_sensors};
pub use units::{Bytes, GigabytesPerSecond, Joules, Percent, Rpm};
