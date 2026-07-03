#[cfg(not(target_vendor = "apple"))]
compile_error!("keisoku supports Apple platforms only (macOS and iOS)");

mod collector;
mod component;
mod device;
mod energy_channel;
mod energy_reading;
mod energy_window;
mod metrics;
mod recorder;
mod sensor;
mod snapshot;
mod units;

mod client;
mod sys;

#[cfg(target_os = "macos")]
mod cf;
#[cfg(target_os = "macos")]
mod cpu_load;
#[cfg(target_os = "macos")]
mod ioreport;
#[cfg(target_os = "macos")]
mod smc;
#[cfg(target_os = "macos")]
mod soc;

pub use collector::Collector;
pub use component::{Component, classify};
pub use device::Device;
pub use energy_channel::EnergyModelChannel;
pub use energy_reading::EnergyReading;
pub use energy_window::EnergyWindow;
pub use metrics::{
    BandwidthMetrics, BatteryMetrics, CpuMetrics, EnergyMetrics, Fan, FanMetrics, GpuMetrics, MemoryMetrics,
    NeuralEngineMetrics, PowerMetrics, Temperatures, ThermalPressure,
};
pub use recorder::{Config, RecorderHandle, Session, start};
pub use sensor::{Sensor, SensorKind, current_sensors, thermal_sensors, voltage_sensors};
pub use snapshot::Snapshot;
pub use units::{Bytes, Celsius, GigabytesPerSecond, Joules, Megahertz, Milliseconds, Percent, Rpm, Watts};

pub fn sensors(kind: SensorKind) -> Box<[Sensor]> {
    client::collect(kind)
}

pub fn sensors_available() -> bool {
    client::is_available()
}
