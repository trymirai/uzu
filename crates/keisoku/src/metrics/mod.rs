//! Derived, serde-serializable metric structs (one per file) plus the memory
//! and thermal-pressure providers.

mod bandwidth;
mod cpu;
mod fans;
mod gpu;
mod memory;
mod neural_engine;
mod power;
mod temperatures;
mod thermal_pressure;

pub use bandwidth::BandwidthMetrics;
pub use cpu::CpuMetrics;
pub use fans::{Fan, FanMetrics};
pub use gpu::GpuMetrics;
pub use memory::MemoryMetrics;
pub(crate) use memory::read as read_memory;
pub use neural_engine::NeuralEngineMetrics;
pub use power::PowerMetrics;
pub use temperatures::Temperatures;
pub use thermal_pressure::ThermalPressure;
pub(crate) use thermal_pressure::read as read_thermal_pressure;
