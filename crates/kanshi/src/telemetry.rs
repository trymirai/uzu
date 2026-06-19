use std::collections::VecDeque;

use keisoku::{Device, Snapshot};

use crate::{disk_row::DiskRow, host_info::HostInfo, net_interface::NetInterface, process_row::ProcessRow};

#[derive(Default)]
pub(crate) struct Telemetry {
    pub(crate) device: Option<Device>,
    pub(crate) host: Option<HostInfo>,
    pub(crate) snapshot: Option<Snapshot>,
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
