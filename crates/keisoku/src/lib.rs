mod collector;
mod component;
mod energy_reading;
mod energy_window;
mod metrics;
mod recorder;
mod sensor;
mod snapshot;
mod units;

#[cfg(target_os = "macos")]
mod cf;
#[cfg(target_vendor = "apple")]
mod client;
#[cfg(target_os = "macos")]
mod cpu_load;
#[cfg(target_os = "macos")]
mod ioreport;
#[cfg(target_os = "macos")]
mod smc;
#[cfg(target_os = "macos")]
mod soc;
#[cfg(target_vendor = "apple")]
mod sys;

pub use collector::Collector;
pub use component::{Component, classify};
pub use energy_reading::EnergyReading;
pub use energy_window::EnergyWindow;
pub use metrics::{
    BandwidthMetrics, BatteryMetrics, CpuMetrics, EnergyMetrics, Fan, FanMetrics, GpuMetrics, MemoryMetrics,
    NeuralEngineMetrics, PowerMetrics, Temperatures, ThermalPressure,
};
pub use recorder::{Config, Device, RecorderHandle, Session, start};
pub use sensor::{Sensor, SensorKind, current_sensors, thermal_sensors, voltage_sensors};
pub use snapshot::Snapshot;
pub use units::{Bytes, Celsius, GigabytesPerSecond, Joules, Megahertz, Milliseconds, Percent, Rpm, Watts};

#[cfg(target_vendor = "apple")]
pub fn sensors(kind: SensorKind) -> Vec<Sensor> {
    client::collect(kind)
}

#[cfg(target_vendor = "apple")]
pub fn sensors_available() -> bool {
    client::is_available()
}

#[cfg(not(target_vendor = "apple"))]
pub fn sensors(_kind: SensorKind) -> Vec<Sensor> {
    Vec::new()
}

#[cfg(not(target_vendor = "apple"))]
pub fn sensors_available() -> bool {
    false
}
