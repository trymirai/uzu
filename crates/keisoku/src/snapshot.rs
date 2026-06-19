use serde::{Deserialize, Serialize};

use crate::{
    metrics::{
        BandwidthMetrics, BatteryMetrics, CpuMetrics, FanMetrics, GpuMetrics, MemoryMetrics, NeuralEngineMetrics,
        PowerMetrics, Temperatures, ThermalPressure,
    },
    sensor::Sensor,
    units::Milliseconds,
};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub elapsed: Milliseconds,
    pub cpu: Option<CpuMetrics>,
    pub gpu: Option<GpuMetrics>,
    pub neural_engine: Option<NeuralEngineMetrics>,
    pub power: Option<PowerMetrics>,
    pub memory: Option<MemoryMetrics>,
    pub bandwidth: Option<BandwidthMetrics>,
    pub fans: Option<FanMetrics>,
    pub battery: Option<BatteryMetrics>,
    pub temperatures: Option<Temperatures>,
    pub thermal_pressure: Option<ThermalPressure>,
    pub sensors: Vec<Sensor>,
}
