use std::collections::VecDeque;

use keisoku::{
    BandwidthMetrics, BatteryMetrics, Bytes, CpuMetrics, FanMetrics, GpuMetrics, MemoryMetrics, NeuralEngineMetrics,
    PowerMetrics, Temperatures, ThermalPressure,
};

use crate::{disk_row::DiskRow, host_info::HostInfo, net_interface::NetInterface, process_row::ProcessRow};

#[derive(Default)]
pub(crate) struct DeviceFacts {
    pub(crate) chip: String,
    pub(crate) efficiency_cores: u8,
    pub(crate) performance_cores: u8,
    pub(crate) gpu_cores: u8,
    pub(crate) ram_total: Bytes,
}

#[derive(Default)]
pub(crate) struct Sample {
    pub(crate) cpu: Option<CpuMetrics>,
    pub(crate) gpu: Option<GpuMetrics>,
    pub(crate) neural_engine: Option<NeuralEngineMetrics>,
    pub(crate) power: Option<PowerMetrics>,
    pub(crate) bandwidth: Option<BandwidthMetrics>,
    pub(crate) memory: Option<MemoryMetrics>,
    pub(crate) fans: Option<FanMetrics>,
    pub(crate) battery: Option<BatteryMetrics>,
    pub(crate) temperatures: Option<Temperatures>,
    pub(crate) thermal_pressure: Option<ThermalPressure>,
}

#[derive(Default)]
pub(crate) struct Telemetry {
    pub(crate) device: Option<DeviceFacts>,
    pub(crate) host: Option<HostInfo>,
    pub(crate) snapshot: Option<Sample>,
    pub(crate) cpu_history: VecDeque<f64>,
    pub(crate) gpu_history: VecDeque<f64>,
    pub(crate) ane_history: VecDeque<f64>,
    pub(crate) memory_history: VecDeque<f64>,
    pub(crate) power_history: VecDeque<f64>,
    pub(crate) max_power: f64,
    pub(crate) uptime_seconds: u64,
    pub(crate) network_down: f64,
    pub(crate) network_up: f64,
    pub(crate) network_packets_down: f64,
    pub(crate) network_packets_up: f64,
    pub(crate) network_total_down: u64,
    pub(crate) network_total_up: u64,
    pub(crate) network_interfaces: Vec<NetInterface>,
    pub(crate) disk_read: f64,
    pub(crate) disk_written: f64,
    pub(crate) disks: Vec<DiskRow>,
    pub(crate) processes: Vec<ProcessRow>,
}
