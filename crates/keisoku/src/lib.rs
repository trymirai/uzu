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
    Constant, Instant, Interval, Session, Static,
    metric::{
        Bandwidth, Battery, Chip, Cons, ConstantMetric, ConstantSet, CpuResidency, CpuUsage, CurrentSensors,
        DramBandwidth, EfficiencyCores, Energy, Fans, GpuCores, GpuUsage, InstantMetric, InstantSet, IntervalFrame,
        IntervalInputs, IntervalMetric, IntervalSet, Memory, Metric, MetricSet, NeuralEngine, Nil, Os,
        PerformanceCores, Power, RailPower, RamTotal, Sample, TemperatureSensors, Temps, Thermal, ValueList, Values,
        VoltageSensors,
    },
    metrics::{
        BandwidthMetrics, BatteryMetrics, CpuMetrics, EnergyMetrics, Fan, FanMetrics, GpuMetrics, MemoryMetrics,
        NeuralEngineMetrics, PowerMetrics, Temperatures, ThermalPressure,
    },
};
pub use sensor::{Sensor, SensorKind, current_sensors, thermal_sensors, voltage_sensors};
pub use sources::Sources;
pub use units::{Bytes, Celsius, GigabytesPerSecond, Joules, Megahertz, Milliseconds, Percent, Rpm, Watts};

pub fn sensors(kind: SensorKind) -> Box<[Sensor]> {
    sources::collect_sensors(kind)
}

pub fn sensors_available() -> bool {
    sources::sensors_available()
}
