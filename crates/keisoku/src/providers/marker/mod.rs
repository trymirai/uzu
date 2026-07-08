mod instant_set;
mod sample;
mod typelist;

mod chip;
mod efficiency_cores;
mod gpu_cores;
mod performance_cores;

mod battery;
mod current_sensors;
mod fans;
mod memory;
mod rail_power;
mod temperature_sensors;
mod thermal;
mod voltage_sensors;

#[cfg(target_os = "macos")]
mod bandwidth;
#[cfg(target_os = "macos")]
mod cpu_usage;
#[cfg(target_os = "macos")]
mod energy;
#[cfg(target_os = "macos")]
mod gpu_usage;
#[cfg(target_os = "macos")]
mod interval_set;
#[cfg(target_os = "macos")]
mod neural_engine;
#[cfg(target_os = "macos")]
mod power;

pub use battery::Battery;
pub use chip::Chip;
pub use current_sensors::CurrentSensors;
pub use efficiency_cores::EfficiencyCores;
pub use fans::Fans;
pub use gpu_cores::GpuCores;
pub use instant_set::{InstantMetric, InstantSet};
pub use memory::Memory;
pub use performance_cores::PerformanceCores;
pub use rail_power::RailPower;
pub use sample::Sample;
pub use temperature_sensors::TemperatureSensors;
pub use thermal::Thermal;
pub use typelist::{Cons, Metric, MetricSet, Nil, ValueList, Values};
pub use voltage_sensors::VoltageSensors;
#[cfg(target_os = "macos")]
pub use {
    crate::sources::interval::{IntervalFrame, IntervalInputs},
    bandwidth::Bandwidth,
    cpu_usage::CpuUsage,
    energy::Energy,
    gpu_usage::GpuUsage,
    interval_set::{IntervalMetric, IntervalSet},
    neural_engine::NeuralEngine,
    power::Power,
};
