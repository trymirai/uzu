#[cfg(not(target_vendor = "apple"))]
compile_error!("keisoku supports Apple platforms only (macOS and iOS)");

mod providers;
mod sources;
mod sys;

mod component;
mod sensor;
mod units;

pub use component::{Component, classify};
pub use providers::{
    Instant, Interval, Session,
    data::{
        BandwidthMetrics, BatteryMetrics, CpuMetrics, EnergyMetrics, Fan, FanMetrics, GpuMetrics, MemoryMetrics,
        NeuralEngineMetrics, PowerMetrics, ThermalPressure,
    },
    marker::{
        Bandwidth, Battery, Chip, Cons, CpuUsage, CurrentSensors, EfficiencyCores, Energy, Fans, GpuCores, GpuUsage,
        InstantMetric, InstantSet, IntervalFrame, IntervalInputs, IntervalMetric, IntervalSet, Memory, Metric,
        MetricSet, NeuralEngine, Nil, PerformanceCores, Power, RailPower, Sample, TemperatureSensors, Thermal,
        ValueList, Values, VoltageSensors,
    },
};
pub use sensor::{Sensor, SensorKind, thermal_sensors};
pub use sources::Sources;
pub use units::{Bytes, Celsius, GigabytesPerSecond, Joules, Megahertz, Percent, Rpm, Watts};

pub fn sensors(kind: SensorKind) -> Box<[Sensor]> {
    sources::collect_sensors(kind)
}

pub fn sensors_available() -> bool {
    sources::sensors_available()
}
