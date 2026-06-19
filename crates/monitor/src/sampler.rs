use std::{
    collections::{HashSet, VecDeque},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use keisoku::Collector;
use sysinfo::{Disks, Networks, ProcessRefreshKind, ProcessesToUpdate, System};

use crate::{
    disk_row::DiskRow,
    host_info::HostInfo,
    net_interface::NetInterface,
    process_row::ProcessRow,
    state::{bump_data_version, interval},
    telemetry::Telemetry,
};

const HISTORY: usize = 256;
const PROCESS_ROWS: usize = 8;
const NETWORK_ROWS: usize = 4;
const DISK_REFRESH: Duration = Duration::from_secs(3);

pub(crate) fn sample_loop(
    telemetry: &Arc<Mutex<Telemetry>>,
    stop: &Arc<AtomicBool>,
) {
    let mut collector = Collector::new();
    let device = collector.device();
    let host = HostInfo {
        user: std::env::var("USER").unwrap_or_else(|_| "user".into()),
        hostname: System::host_name().unwrap_or_default(),
        os: System::long_os_version().unwrap_or_default(),
        kernel: System::kernel_version().unwrap_or_default(),
        shell: std::env::var("SHELL")
            .ok()
            .and_then(|path| path.rsplit('/').next().map(str::to_owned))
            .unwrap_or_default(),
    };
    if let Ok(mut state) = telemetry.lock() {
        state.device = Some(device);
        state.host = Some(host);
    }

    let mut system = System::new();
    let mut networks = Networks::new_with_refreshed_list();
    let mut all_disks = Disks::new_with_refreshed_list();
    let mut last_refresh = Instant::now();
    let mut last_disk_refresh: Option<Instant> = None;

    let process_kind = ProcessRefreshKind::nothing().with_cpu().with_memory().with_disk_usage();

    while !stop.load(Ordering::Relaxed) {
        let snapshot = collector.sample(interval());

        system.refresh_processes_specifics(ProcessesToUpdate::All, true, process_kind);
        networks.refresh(true);
        let elapsed = last_refresh.elapsed().as_secs_f64().max(0.001);
        last_refresh = Instant::now();

        let (mut received, mut transmitted) = (0u64, 0u64);
        let (mut packets_in, mut packets_out) = (0u64, 0u64);
        let (mut total_in, mut total_out) = (0u64, 0u64);
        let mut interfaces: Vec<NetInterface> = Vec::new();
        for (name, data) in networks.iter() {
            received += data.received();
            transmitted += data.transmitted();
            packets_in += data.packets_received();
            packets_out += data.packets_transmitted();
            total_in += data.total_received();
            total_out += data.total_transmitted();

            if data.total_received() > 0 || data.total_transmitted() > 0 {
                interfaces.push(NetInterface {
                    name: name.clone(),
                    down: data.received() as f64 / elapsed,
                    up: data.transmitted() as f64 / elapsed,
                });
            }
        }
        interfaces.sort_by(|a, b| (b.down + b.up).partial_cmp(&(a.down + a.up)).unwrap_or(std::cmp::Ordering::Equal));
        interfaces.truncate(NETWORK_ROWS);

        let disks = last_disk_refresh.is_none_or(|at| at.elapsed() >= DISK_REFRESH).then(|| {
            last_disk_refresh = Some(Instant::now());
            all_disks.refresh(true);
            let mut seen_disks = HashSet::new();
            all_disks
                .list()
                .iter()
                .filter(|disk| seen_disks.insert(disk.name().to_os_string()))
                .map(|disk| DiskRow {
                    name: disk.name().to_string_lossy().into_owned(),
                    used: disk.total_space().saturating_sub(disk.available_space()),
                    total: disk.total_space(),
                })
                .collect::<Vec<_>>()
        });

        let (mut disk_read, mut disk_written) = (0u64, 0u64);
        for process in system.processes().values() {
            let io = process.disk_usage();
            disk_read += io.read_bytes;
            disk_written += io.written_bytes;
        }

        let mut processes = system
            .processes()
            .values()
            .map(|process| ProcessRow {
                cpu: process.cpu_usage(),
                memory: process.memory(),
                name: process.name().to_string_lossy().into_owned(),
            })
            .collect::<Vec<_>>();
        processes.sort_by(|a, b| b.cpu.partial_cmp(&a.cpu).unwrap_or(std::cmp::Ordering::Equal));
        processes.truncate(PROCESS_ROWS);

        let cpu = snapshot.cpu.as_ref().map(|c| c.usage.value() as f64).unwrap_or(0.0);
        let gpu = snapshot.gpu.as_ref().map(|g| g.usage.value() as f64).unwrap_or(0.0);
        let ane = snapshot.neural_engine.as_ref().map(|a| a.active.value() as f64).unwrap_or(0.0);
        let memory = snapshot
            .memory
            .as_ref()
            .filter(|m| m.ram_total.value() > 0)
            .map(|m| m.ram_usage.value() as f64 / m.ram_total.value() as f64 * 100.0)
            .unwrap_or(0.0);

        let power = snapshot.power.as_ref().map(|p| p.package.value() as f64).unwrap_or(0.0);

        if let Ok(mut state) = telemetry.lock() {
            push_history(&mut state.cpu_history, cpu);
            push_history(&mut state.gpu_history, gpu);
            push_history(&mut state.ane_history, ane);
            push_history(&mut state.memory_history, memory);
            push_history(&mut state.power_history, power);
            state.max_power = state.max_power.max(power);
            state.snapshot = Some(snapshot);
            state.uptime_seconds = System::uptime();
            state.network_down = received as f64 / elapsed;
            state.network_up = transmitted as f64 / elapsed;
            state.network_packets_down = packets_in as f64 / elapsed;
            state.network_packets_up = packets_out as f64 / elapsed;
            state.network_total_down = total_in;
            state.network_total_up = total_out;
            state.network_interfaces = interfaces;
            state.disk_read = disk_read as f64 / elapsed;
            state.disk_written = disk_written as f64 / elapsed;
            if let Some(disks) = disks {
                state.disks = disks;
            }
            state.processes = processes;
        }
        bump_data_version();
    }
}

fn push_history(
    history: &mut VecDeque<f64>,
    value: f64,
) {
    history.push_back(value);
    while history.len() > HISTORY {
        history.pop_front();
    }
}
