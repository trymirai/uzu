#[cfg(target_os = "macos")]
use super::IntervalHandle;
#[cfg(target_os = "macos")]
use crate::marker::IntervalSet;
use crate::{
    metrics::{BatteryMetrics, FanMetrics, MemoryMetrics, ThermalPressure},
    sensor::Sensor,
    sources::Sources,
    units::Watts,
};

/// Instantaneous device facts and gauges (chip, memory, sensors, …).
pub struct Device {
    sources: Sources,
}

impl Device {
    pub fn new() -> Self {
        Self {
            sources: Sources::new(),
        }
    }

    pub fn os_version(&self) -> String {
        self.sources.os_version()
    }

    pub fn chip(&self) -> String {
        self.sources.chip()
    }

    pub fn efficiency_cores(&self) -> u8 {
        self.sources.efficiency_cores()
    }

    pub fn performance_cores(&self) -> u8 {
        self.sources.performance_cores()
    }

    pub fn gpu_cores(&self) -> u8 {
        self.sources.gpu_cores()
    }

    pub fn memory(&mut self) -> Option<MemoryMetrics> {
        self.sources.memory()
    }

    pub fn battery(&mut self) -> Option<BatteryMetrics> {
        self.sources.battery()
    }

    pub fn thermal(&mut self) -> Option<ThermalPressure> {
        self.sources.thermal()
    }

    pub fn fans(&self) -> Option<FanMetrics> {
        self.sources.fans()
    }

    pub fn temperature_sensors(&mut self) -> Box<[Sensor]> {
        self.sources.temperature_sensors()
    }

    pub fn voltage_sensors(&mut self) -> Box<[Sensor]> {
        self.sources.voltage_sensors()
    }

    pub fn current_sensors(&mut self) -> Box<[Sensor]> {
        self.sources.current_sensors()
    }

    pub fn rail_power(&mut self) -> Option<Watts> {
        self.sources.rail_power()
    }

    /// Starts an IOReport interval measurement for the channels in `M`.
    ///
    /// Call [`IntervalHandle::start`] before work and [`IntervalHandle::stop`] after.
    #[cfg(target_os = "macos")]
    pub fn interval_measurement<M: IntervalSet>() -> IntervalHandle<M> {
        IntervalHandle::new()
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::new()
    }
}
