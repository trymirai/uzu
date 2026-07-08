#[cfg(not(target_vendor = "apple"))]
compile_error!("keisoku supports Apple platforms only (macOS and iOS)");

mod providers;
mod sources;
mod sys;

mod component;
mod power_meter;
mod sensor;
mod units;

pub use component::{Component, classify};
pub use power_meter::{PowerMeter, PowerReading};
pub use providers::{
    Instant,
    data::{
        BandwidthMetrics, BatteryMetrics, CpuMetrics, EnergyMetrics, Fan, FanMetrics, GpuMetrics, MemoryMetrics,
        NeuralEngineMetrics, PowerMetrics, ThermalPressure,
    },
    marker::{
        Battery, Chip, Cons, CurrentSensors, EfficiencyCores, Fans, GpuCores, InstantMetric, InstantSet, Memory,
        Metric, MetricSet, Nil, PerformanceCores, RailPower, Sample, TemperatureSensors, Thermal, ValueList, Values,
        VoltageSensors,
    },
};
#[cfg(target_os = "macos")]
pub use providers::{
    Interval, Session,
    marker::{
        Bandwidth, CpuUsage, Energy, GpuUsage, IntervalFrame, IntervalInputs, IntervalMetric, IntervalSet,
        NeuralEngine, Power,
    },
};
pub use sensor::{Sensor, SensorKind, thermal_sensors};
pub use sources::Sources;
pub use units::{Bytes, GigabytesPerSecond, Joules, Megahertz, Percent, Rpm, Watts};
