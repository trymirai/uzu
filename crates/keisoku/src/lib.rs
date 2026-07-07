#[cfg(not(target_vendor = "apple"))]
compile_error!("keisoku supports Apple platforms only (macOS and iOS)");

mod component;
mod decode;
mod metric;
mod metrics;
mod provider;
mod sensor;
mod sources;
mod units;

mod client;
mod sys;

#[cfg(target_os = "macos")]
mod cf;
#[cfg(target_os = "macos")]
mod ioreport;
#[cfg(target_os = "macos")]
mod smc;
#[cfg(target_os = "macos")]
mod soc;

pub use component::{Component, classify};
pub use decode::{EnergyTotals, FrequencyTables, RawChannel};
pub use metric::{
    Bandwidth, Battery, Chip, CpuResidency, CpuUsage, CurrentSensors, DramBandwidth, EfficiencyCores, Energy, Fans,
    GpuCores, GpuUsage, IoReportGroups, Measured, Memory, NeuralEngine, Os, PackageWatts, PerformanceCores, Power,
    RailPower, RamTotal, Reading, TemperatureSensors, Temps, Thermal, VoltageSensors,
};
pub use metrics::{
    BandwidthMetrics, BatteryMetrics, CpuMetrics, EnergyMetrics, Fan, FanMetrics, GpuMetrics, MemoryMetrics,
    NeuralEngineMetrics, PowerMetrics, Temperatures, ThermalPressure,
};
pub use provider::{Instant, Interval, Session, Static};
pub use sensor::{Sensor, SensorKind, current_sensors, thermal_sensors, voltage_sensors};
pub use sources::Sources;
pub use units::{Bytes, Celsius, GigabytesPerSecond, Joules, Megahertz, Milliseconds, Percent, Rpm, Watts};

pub fn sensors(kind: SensorKind) -> Box<[Sensor]> {
    client::collect(kind)
}

pub fn sensors_available() -> bool {
    client::is_available()
}
