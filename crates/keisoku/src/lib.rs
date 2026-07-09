#[cfg(not(target_vendor = "apple"))]
compile_error!("keisoku supports Apple platforms only (macOS and iOS)");

mod device;
mod marker;
mod metrics;
mod sources;
mod sys;

mod component;
mod power_meter;
mod sensor;
mod units;

pub use component::{Component, classify};
pub use device::Device;
#[cfg(target_os = "macos")]
pub use device::IntervalHandle;
#[cfg(target_os = "macos")]
pub use marker::{
    Ane, AneBandwidth, ChannelMetric, ChannelSet, Cons, Cpu, DramBytes, DramHistogram, DramRead, DramWrite, EnergyRail,
    Gpu, IntervalSet, Nil, Ram, Sample,
};
pub use metrics::{BatteryMetrics, Fan, FanMetrics, MemoryMetrics, ThermalPressure};
pub use power_meter::{PowerMeter, PowerReading};
pub use sensor::{Sensor, SensorKind, thermal_sensors};
#[cfg(target_os = "macos")]
pub use sys::ioreport::{
    decode::Channel,
    kinds::{DramFlow, Rail},
};
pub use units::{Bytes, GigabytesPerSecond, Joules, Megahertz, Percent, Rpm, Watts};
