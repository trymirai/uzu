use bitflags::bitflags;

use crate::{
    component::Component,
    decode,
    metrics::{
        BandwidthMetrics, BatteryMetrics, CpuMetrics, EnergyMetrics, FanMetrics, GpuMetrics, MemoryMetrics,
        NeuralEngineMetrics, PowerMetrics, Temperatures, ThermalPressure,
    },
    provider::Window,
    sensor::Sensor,
    sources::Sources,
    units::{Bytes, Celsius, GigabytesPerSecond, Joules, Megahertz, Percent, Watts},
};

pub trait Reading {
    type Value;
    fn read(sources: &mut Sources) -> Self::Value;
}

pub trait Measured {
    type Value;
    const GROUPS: IoReportGroups;
    fn extract(window: &Window) -> Self::Value;
}

bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct IoReportGroups: u8 {
        const ENERGY_MODEL = 1 << 0;
        const CPU_STATS = 1 << 1;
        const GPU_STATS = 1 << 2;
        const AMC_STATS = 1 << 3;
        const PMP = 1 << 4;
    }
}

pub struct Os;

impl Reading for Os {
    type Value = String;

    fn read(_sources: &mut Sources) -> String {
        sysinfo::System::long_os_version().unwrap_or_default()
    }
}

pub struct Chip;

impl Reading for Chip {
    type Value = String;

    fn read(sources: &mut Sources) -> String {
        #[cfg(target_os = "macos")]
        if let Some(soc) = sources.soc()
            && !soc.chip_name.is_empty()
        {
            return soc.chip_name.clone();
        }
        #[cfg(not(target_os = "macos"))]
        if let Some(model) = sysctl_string("hw.machine").filter(|model| !model.is_empty()) {
            return model;
        }
        sources.system().cpus().first().map(|cpu| cpu.brand().trim().to_string()).unwrap_or_default()
    }
}

pub struct RamTotal;

impl Reading for RamTotal {
    type Value = Bytes;

    fn read(sources: &mut Sources) -> Bytes {
        Bytes(sources.system().total_memory())
    }
}

pub struct EfficiencyCores;

impl Reading for EfficiencyCores {
    type Value = u8;

    fn read(sources: &mut Sources) -> u8 {
        #[cfg(target_os = "macos")]
        {
            sources.soc().map(|soc| soc.ecpu_cores).unwrap_or(0)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = sources;
            perflevel_cores().1
        }
    }
}

pub struct PerformanceCores;

impl Reading for PerformanceCores {
    type Value = u8;

    fn read(sources: &mut Sources) -> u8 {
        #[cfg(target_os = "macos")]
        {
            sources.soc().map(|soc| soc.pcpu_cores).unwrap_or(0)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = sources;
            perflevel_cores().0
        }
    }
}

pub struct GpuCores;

impl Reading for GpuCores {
    type Value = u8;

    fn read(sources: &mut Sources) -> u8 {
        #[cfg(target_os = "macos")]
        {
            sources.soc().map(|soc| soc.gpu_cores).unwrap_or(0)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = sources;
            0
        }
    }
}

pub struct Memory;

impl Reading for Memory {
    type Value = Option<MemoryMetrics>;

    fn read(_sources: &mut Sources) -> Option<MemoryMetrics> {
        MemoryMetrics::read()
    }
}

pub struct Battery;

impl Reading for Battery {
    type Value = Option<BatteryMetrics>;

    fn read(_sources: &mut Sources) -> Option<BatteryMetrics> {
        BatteryMetrics::read()
    }
}

pub struct Temps;

impl Reading for Temps {
    type Value = Option<Temperatures>;

    fn read(sources: &mut Sources) -> Option<Temperatures> {
        let sensors = sources.temperature_sensors();
        (!sensors.is_empty()).then(|| temperatures_from(&sensors))
    }
}

pub struct Fans;

impl Reading for Fans {
    type Value = Option<FanMetrics>;

    fn read(sources: &mut Sources) -> Option<FanMetrics> {
        #[cfg(target_os = "macos")]
        {
            sources.smc().map(|smc| smc.fans())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = sources;
            None
        }
    }
}

pub struct Thermal;

impl Reading for Thermal {
    type Value = Option<ThermalPressure>;

    fn read(_sources: &mut Sources) -> Option<ThermalPressure> {
        ThermalPressure::read()
    }
}

pub struct PackageWatts;

impl Reading for PackageWatts {
    type Value = Option<Watts>;

    fn read(sources: &mut Sources) -> Option<Watts> {
        #[cfg(target_os = "macos")]
        {
            sources.smc().and_then(|smc| smc.package_watts())
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = sources;
            None
        }
    }
}

pub struct TemperatureSensors;

impl Reading for TemperatureSensors {
    type Value = Box<[Sensor]>;

    fn read(sources: &mut Sources) -> Box<[Sensor]> {
        sources.temperature_sensors()
    }
}

pub struct VoltageSensors;

impl Reading for VoltageSensors {
    type Value = Box<[Sensor]>;

    fn read(sources: &mut Sources) -> Box<[Sensor]> {
        sources.voltage_sensors()
    }
}

pub struct CurrentSensors;

impl Reading for CurrentSensors {
    type Value = Box<[Sensor]>;

    fn read(sources: &mut Sources) -> Box<[Sensor]> {
        sources.current_sensors()
    }
}

pub struct RailPower;

impl Reading for RailPower {
    type Value = Option<Watts>;

    fn read(sources: &mut Sources) -> Option<Watts> {
        let voltage = sources.voltage_sensors();
        let current = sources.current_sensors();
        rail_power(&voltage, &current)
    }
}

pub struct CpuUsage;

impl Measured for CpuUsage {
    type Value = CpuMetrics;
    const GROUPS: IoReportGroups = IoReportGroups::CPU_STATS;

    fn extract(window: &Window) -> CpuMetrics {
        let frequencies = window.frequencies;
        let (ecpu, pcpu) = decode::cpu_clusters(window.channels, frequencies.ecpu, frequencies.pcpu);
        let efficiency_cores = frequencies.ecpu_cores as f32;
        let performance_cores = frequencies.pcpu_cores as f32;
        let usage = decode::divide_or_zero(
            ecpu.1 * efficiency_cores + pcpu.1 * performance_cores,
            efficiency_cores + performance_cores,
        );
        CpuMetrics {
            usage: Percent(usage * 100.0),
            ecpu_frequency: Megahertz(ecpu.0),
            pcpu_frequency: Megahertz(pcpu.0),
        }
    }
}

pub struct GpuUsage;

impl Measured for GpuUsage {
    type Value = GpuMetrics;
    const GROUPS: IoReportGroups = IoReportGroups::GPU_STATS;

    fn extract(window: &Window) -> GpuMetrics {
        let (frequency, usage) = decode::gpu_frequency(window.channels, window.frequencies.gpu);
        GpuMetrics {
            frequency: Megahertz(frequency),
            usage: Percent(usage * 100.0),
        }
    }
}

pub struct NeuralEngine;

impl Measured for NeuralEngine {
    type Value = NeuralEngineMetrics;
    const GROUPS: IoReportGroups = IoReportGroups::PMP;

    fn extract(window: &Window) -> NeuralEngineMetrics {
        NeuralEngineMetrics {
            active: Percent(decode::ane_active_percent(window.channels)),
        }
    }
}

pub struct Power;

impl Measured for Power {
    type Value = PowerMetrics;
    const GROUPS: IoReportGroups = IoReportGroups::ENERGY_MODEL;

    fn extract(window: &Window) -> PowerMetrics {
        let energy = decode::energy_totals(window.channels);
        let seconds = window.elapsed.as_secs_f64().max(0.001);
        let package = window.package_watts_mean.map(Watts).unwrap_or_else(|| Watts((energy.total() / seconds) as f32));
        PowerMetrics {
            cpu: Watts((energy.cpu / seconds) as f32),
            gpu: Watts((energy.gpu / seconds) as f32),
            ane: Watts((energy.ane / seconds) as f32),
            ram: Watts((energy.ram / seconds) as f32),
            package,
        }
    }
}

pub struct Energy;

impl Measured for Energy {
    type Value = EnergyMetrics;
    const GROUPS: IoReportGroups = IoReportGroups::ENERGY_MODEL;

    fn extract(window: &Window) -> EnergyMetrics {
        let energy = decode::energy_totals(window.channels);
        let seconds = window.elapsed.as_secs_f32().max(0.001);
        let package =
            window.package_watts_mean.map(|watts| Joules(watts * seconds)).unwrap_or(Joules(energy.total() as f32));
        EnergyMetrics {
            cpu: Joules(energy.cpu as f32),
            gpu: Joules(energy.gpu as f32),
            ane: Joules(energy.ane as f32),
            ram: Joules(energy.ram as f32),
            package,
        }
    }
}

pub struct Bandwidth;

impl Measured for Bandwidth {
    type Value = BandwidthMetrics;
    const GROUPS: IoReportGroups = IoReportGroups::AMC_STATS.union(IoReportGroups::PMP);

    fn extract(window: &Window) -> BandwidthMetrics {
        let window_milliseconds = window.elapsed.as_millis().max(1) as u64;
        let (read, write) = decode::dram_bandwidth(window.channels, window_milliseconds);
        BandwidthMetrics {
            dram_read: GigabytesPerSecond(read),
            dram_write: GigabytesPerSecond(write),
        }
    }
}

macro_rules! tuple_impls {
    ($($member:ident),+) => {
        impl<$($member: Reading),+> Reading for ($($member,)+) {
            type Value = ($($member::Value,)+);

            fn read(sources: &mut Sources) -> Self::Value {
                ($($member::read(sources),)+)
            }
        }

        impl<$($member: Measured),+> Measured for ($($member,)+) {
            type Value = ($($member::Value,)+);
            const GROUPS: IoReportGroups = IoReportGroups::empty()$(.union($member::GROUPS))+;

            fn extract(window: &Window) -> Self::Value {
                ($($member::extract(window),)+)
            }
        }
    };
}

tuple_impls!(A);
tuple_impls!(A, B);
tuple_impls!(A, B, C);
tuple_impls!(A, B, C, D);
tuple_impls!(A, B, C, D, E);
tuple_impls!(A, B, C, D, E, F);
tuple_impls!(A, B, C, D, E, F, G);
tuple_impls!(A, B, C, D, E, F, G, H);
tuple_impls!(A, B, C, D, E, F, G, H, I);
tuple_impls!(A, B, C, D, E, F, G, H, I, J);
tuple_impls!(A, B, C, D, E, F, G, H, I, J, K);
tuple_impls!(A, B, C, D, E, F, G, H, I, J, K, L);

impl Reading for () {
    type Value = ();

    fn read(_sources: &mut Sources) {}
}

impl Measured for () {
    type Value = ();
    const GROUPS: IoReportGroups = IoReportGroups::empty();

    fn extract(_window: &Window) {}
}

fn temperatures_from(sensors: &[Sensor]) -> Temperatures {
    let average_of = |components: &[Component]| {
        let values: Vec<f32> = sensors
            .iter()
            .filter(|sensor| components.contains(&sensor.component) && (1.0..150.0).contains(&sensor.value))
            .map(|sensor| sensor.value as f32)
            .collect();
        (!values.is_empty()).then(|| Celsius(average(&values)))
    };
    Temperatures {
        cpu_average: average_of(&[Component::Cpu, Component::Soc]),
        gpu_average: average_of(&[Component::Gpu]),
    }
}

fn average(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn rail_power(
    voltage: &[Sensor],
    current: &[Sensor],
) -> Option<Watts> {
    const MAX_PLAUSIBLE_WATTS: f64 = 1000.0;
    let split_area_code = |name: &str| name.rsplit_once(' ').map(|(area, code)| (area.to_owned(), code.to_owned()));
    let is_battery_rail = |sensor: &&Sensor| matches!(sensor.component, Component::Charger | Component::Battery);

    let mut total_watts = 0f64;
    for voltage_sensor in voltage.iter().filter(is_battery_rail) {
        let Some((voltage_area, voltage_code)) = split_area_code(&voltage_sensor.name) else {
            continue;
        };
        let Some(rail_code) = voltage_code.strip_prefix('V').filter(|code| !code.is_empty()) else {
            continue;
        };
        for current_sensor in current.iter().filter(is_battery_rail) {
            let Some((current_area, current_code)) = split_area_code(&current_sensor.name) else {
                continue;
            };
            if current_area == voltage_area && current_code.strip_prefix('I') == Some(rail_code) {
                let watts = (voltage_sensor.value * current_sensor.value).abs();
                if (0.0..=MAX_PLAUSIBLE_WATTS).contains(&watts) {
                    total_watts += watts;
                }
            }
        }
    }
    (total_watts > 0.0).then_some(Watts(total_watts as f32))
}

#[cfg(not(target_os = "macos"))]
fn sysctl_string(name: &str) -> Option<String> {
    let name = std::ffi::CString::new(name).ok()?;
    let mut len = 0usize;
    let probe = unsafe { libc::sysctlbyname(name.as_ptr(), std::ptr::null_mut(), &mut len, std::ptr::null_mut(), 0) };
    if probe != 0 || len == 0 {
        return None;
    }
    let mut buffer = vec![0u8; len];
    let read =
        unsafe { libc::sysctlbyname(name.as_ptr(), buffer.as_mut_ptr().cast(), &mut len, std::ptr::null_mut(), 0) };
    if read != 0 {
        return None;
    }
    if let Some(nul) = buffer.iter().position(|&byte| byte == 0) {
        buffer.truncate(nul);
    }
    String::from_utf8(buffer).ok()
}

#[cfg(not(target_os = "macos"))]
fn sysctl_u32(name: &str) -> Option<u32> {
    let name = std::ffi::CString::new(name).ok()?;
    let mut value = 0u32;
    let mut len = std::mem::size_of::<u32>();
    let read = unsafe {
        libc::sysctlbyname(name.as_ptr(), std::ptr::addr_of_mut!(value).cast(), &mut len, std::ptr::null_mut(), 0)
    };
    (read == 0).then_some(value)
}

#[cfg(not(target_os = "macos"))]
fn perflevel_cores() -> (u8, u8) {
    let performance = sysctl_u32("hw.perflevel0.logicalcpu").unwrap_or(0);
    let efficiency = if sysctl_u32("hw.nperflevels").unwrap_or(1) > 1 {
        sysctl_u32("hw.perflevel1.logicalcpu").unwrap_or(0)
    } else {
        0
    };
    (performance as u8, efficiency as u8)
}
