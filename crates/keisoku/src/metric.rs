use std::time::Duration;

use bitflags::bitflags;
use obfstr::obfstr;

use crate::{
    component::Component,
    decode::{self, EnergyTotals, FrequencyTables, GroupId, RawChannel},
    metrics::{
        BandwidthMetrics, BatteryMetrics, CpuMetrics, EnergyMetrics, FanMetrics, GpuMetrics, MemoryMetrics,
        NeuralEngineMetrics, PowerMetrics, Temperatures, ThermalPressure,
    },
    sensor::Sensor,
    sources::Sources,
    units::{Bytes, Celsius, GigabytesPerSecond, Joules, Megahertz, Percent, Watts},
};

/// Static + Instant metrics: every value meaningful from a single read.
pub trait Reading {
    type Value;
    fn read(sources: &mut Sources) -> Self::Value;
}

/// Interval metrics. Each declares exactly the IOReport channels it subscribes
/// to (`GROUPS`), the non-channel inputs it needs (`Ctx`), and how it folds the
/// window's channels into its value (`Acc` + `consume`/`finish`). Nothing shared
/// or unused is passed in — an energy-only interval never sees frequency tables.
pub trait Measured {
    type Value;
    /// Exactly this metric's non-channel inputs (frequency tables, package watts, …).
    type Ctx<'a>;
    /// Per-metric fold state, seeded before the single channel pass.
    type Acc: Default;
    const GROUPS: IoReportGroups;

    fn context(
        sources: &Sources,
        package_watts_mean: Option<f32>,
    ) -> Self::Ctx<'_>;
    fn consume(
        acc: &mut Self::Acc,
        channel: &RawChannel,
        ctx: &Self::Ctx<'_>,
    );
    fn finish(
        acc: Self::Acc,
        elapsed: Duration,
        ctx: &Self::Ctx<'_>,
    ) -> Self::Value;
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

#[derive(Default)]
pub struct CpuResidency {
    ecpu: Vec<(u32, f32)>,
    pcpu: Vec<(u32, f32)>,
}

impl Measured for CpuUsage {
    type Value = CpuMetrics;
    type Ctx<'a> = FrequencyTables<'a>;
    type Acc = CpuResidency;
    const GROUPS: IoReportGroups = IoReportGroups::CPU_STATS;

    fn context(
        sources: &Sources,
        _package_watts_mean: Option<f32>,
    ) -> FrequencyTables<'_> {
        #[cfg(target_os = "macos")]
        {
            sources.frequencies()
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = sources;
            FrequencyTables::default()
        }
    }

    fn consume(
        acc: &mut CpuResidency,
        channel: &RawChannel,
        ctx: &FrequencyTables<'_>,
    ) {
        if channel.group == GroupId::CpuStats && channel.subgroup == obfstr!("CPU Core Performance States") {
            if channel.name.starts_with(obfstr!("PCPU")) {
                acc.pcpu.push(decode::calculate_frequency(&channel.states, ctx.pcpu));
            } else if channel.name.starts_with(obfstr!("ECPU")) || channel.name.starts_with(obfstr!("MCPU")) {
                acc.ecpu.push(decode::calculate_frequency(&channel.states, ctx.ecpu));
            }
        }
    }

    fn finish(
        acc: CpuResidency,
        _elapsed: Duration,
        ctx: &FrequencyTables<'_>,
    ) -> CpuMetrics {
        let ecpu_readings: Vec<(u32, f32)> = acc.ecpu.iter().copied().filter(|&(_, percent)| percent > 0.0).collect();
        let ecpu = decode::average_cluster_frequency(&ecpu_readings, ctx.ecpu);
        let pcpu = decode::average_cluster_frequency(&acc.pcpu, ctx.pcpu);
        let efficiency_cores = ctx.ecpu_cores as f32;
        let performance_cores = ctx.pcpu_cores as f32;
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
    type Ctx<'a> = FrequencyTables<'a>;
    type Acc = (u32, f32);
    const GROUPS: IoReportGroups = IoReportGroups::GPU_STATS;

    fn context(
        sources: &Sources,
        _package_watts_mean: Option<f32>,
    ) -> FrequencyTables<'_> {
        #[cfg(target_os = "macos")]
        {
            sources.frequencies()
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = sources;
            FrequencyTables::default()
        }
    }

    fn consume(
        acc: &mut (u32, f32),
        channel: &RawChannel,
        ctx: &FrequencyTables<'_>,
    ) {
        if channel.group == GroupId::GpuStats
            && channel.subgroup == obfstr!("GPU Performance States")
            && channel.name == obfstr!("GPUPH")
            && ctx.gpu.len() > 1
        {
            *acc = decode::calculate_frequency(&channel.states, &ctx.gpu[1..]);
        }
    }

    fn finish(
        acc: (u32, f32),
        _elapsed: Duration,
        _ctx: &FrequencyTables<'_>,
    ) -> GpuMetrics {
        GpuMetrics {
            frequency: Megahertz(acc.0),
            usage: Percent(acc.1 * 100.0),
        }
    }
}

pub struct NeuralEngine;

impl Measured for NeuralEngine {
    type Value = NeuralEngineMetrics;
    type Ctx<'a> = ();
    type Acc = f32;
    const GROUPS: IoReportGroups = IoReportGroups::PMP;

    fn context(
        _sources: &Sources,
        _package_watts_mean: Option<f32>,
    ) {
    }

    fn consume(
        acc: &mut f32,
        channel: &RawChannel,
        _ctx: &(),
    ) {
        if channel.group == GroupId::Pmp
            && channel.subgroup.contains(obfstr!("Floor"))
            && (channel.name == obfstr!("ANE-AF-BW") || channel.name == obfstr!("ANE-DCS-BW"))
        {
            *acc = acc.max(decode::residency_active_percent(&channel.states));
        }
    }

    fn finish(
        acc: f32,
        _elapsed: Duration,
        _ctx: &(),
    ) -> NeuralEngineMetrics {
        NeuralEngineMetrics {
            active: Percent(acc),
        }
    }
}

pub struct Power;

impl Measured for Power {
    type Value = PowerMetrics;
    type Ctx<'a> = Option<f32>;
    type Acc = EnergyTotals;
    const GROUPS: IoReportGroups = IoReportGroups::ENERGY_MODEL;

    fn context(
        _sources: &Sources,
        package_watts_mean: Option<f32>,
    ) -> Option<f32> {
        package_watts_mean
    }

    fn consume(
        acc: &mut EnergyTotals,
        channel: &RawChannel,
        _ctx: &Option<f32>,
    ) {
        if channel.group == GroupId::EnergyModel {
            acc.accumulate(&channel.name, channel.integer_value, &channel.unit);
        }
    }

    fn finish(
        acc: EnergyTotals,
        elapsed: Duration,
        ctx: &Option<f32>,
    ) -> PowerMetrics {
        let seconds = elapsed.as_secs_f64().max(0.001);
        let package = ctx.map(Watts).unwrap_or_else(|| Watts((acc.total() / seconds) as f32));
        PowerMetrics {
            cpu: Watts((acc.cpu / seconds) as f32),
            gpu: Watts((acc.gpu / seconds) as f32),
            ane: Watts((acc.ane / seconds) as f32),
            ram: Watts((acc.ram / seconds) as f32),
            package,
        }
    }
}

pub struct Energy;

impl Measured for Energy {
    type Value = EnergyMetrics;
    type Ctx<'a> = Option<f32>;
    type Acc = EnergyTotals;
    const GROUPS: IoReportGroups = IoReportGroups::ENERGY_MODEL;

    fn context(
        _sources: &Sources,
        package_watts_mean: Option<f32>,
    ) -> Option<f32> {
        package_watts_mean
    }

    fn consume(
        acc: &mut EnergyTotals,
        channel: &RawChannel,
        _ctx: &Option<f32>,
    ) {
        if channel.group == GroupId::EnergyModel {
            acc.accumulate(&channel.name, channel.integer_value, &channel.unit);
        }
    }

    fn finish(
        acc: EnergyTotals,
        elapsed: Duration,
        ctx: &Option<f32>,
    ) -> EnergyMetrics {
        let seconds = elapsed.as_secs_f32().max(0.001);
        let package = ctx.map(|watts| Joules(watts * seconds)).unwrap_or(Joules(acc.total() as f32));
        EnergyMetrics {
            cpu: Joules(acc.cpu as f32),
            gpu: Joules(acc.gpu as f32),
            ane: Joules(acc.ane as f32),
            ram: Joules(acc.ram as f32),
            package,
        }
    }
}

pub struct Bandwidth;

#[derive(Default)]
pub struct DramBandwidth {
    read_bytes: f64,
    write_bytes: f64,
    read_histogram: f32,
    write_histogram: f32,
}

impl Measured for Bandwidth {
    type Value = BandwidthMetrics;
    type Ctx<'a> = ();
    type Acc = DramBandwidth;
    const GROUPS: IoReportGroups = IoReportGroups::AMC_STATS.union(IoReportGroups::PMP);

    fn context(
        _sources: &Sources,
        _package_watts_mean: Option<f32>,
    ) {
    }

    fn consume(
        acc: &mut DramBandwidth,
        channel: &RawChannel,
        _ctx: &(),
    ) {
        if channel.group == GroupId::AmcStats {
            let bytes = channel.integer_value as f64;
            if bytes > 0.0 {
                let aggregate = decode::strip_die_prefix(&channel.name);
                if aggregate == obfstr!("DCS RD") {
                    acc.read_bytes += bytes;
                } else if aggregate == obfstr!("DCS WR") {
                    acc.write_bytes += bytes;
                }
            }
        } else if channel.group == GroupId::Pmp && channel.subgroup == obfstr!("DRAM BW") {
            let gbps = decode::residency_weighted_gbps(&channel.states);
            match decode::dram_flow(&channel.name) {
                Some(true) => acc.read_histogram = acc.read_histogram.max(gbps),
                Some(false) => acc.write_histogram = acc.write_histogram.max(gbps),
                None => {},
            }
        }
    }

    fn finish(
        acc: DramBandwidth,
        elapsed: Duration,
        _ctx: &(),
    ) -> BandwidthMetrics {
        let seconds = elapsed.as_secs_f64().max(0.001);
        let to_gbps = |bytes: f64| (bytes / seconds / 1e9) as f32;
        let read = if acc.read_bytes > 0.0 {
            to_gbps(acc.read_bytes)
        } else {
            acc.read_histogram
        };
        let write = if acc.write_bytes > 0.0 {
            to_gbps(acc.write_bytes)
        } else {
            acc.write_histogram
        };
        BandwidthMetrics {
            dram_read: GigabytesPerSecond(read),
            dram_write: GigabytesPerSecond(write),
        }
    }
}

macro_rules! tuple_impls {
    ($($member:ident $index:tt),+) => {
        impl<$($member: Reading),+> Reading for ($($member,)+) {
            type Value = ($($member::Value,)+);

            fn read(sources: &mut Sources) -> Self::Value {
                ($($member::read(sources),)+)
            }
        }

        impl<$($member: Measured),+> Measured for ($($member,)+) {
            type Value = ($($member::Value,)+);
            type Ctx<'a> = ($($member::Ctx<'a>,)+);
            type Acc = ($($member::Acc,)+);
            const GROUPS: IoReportGroups = IoReportGroups::empty()$(.union($member::GROUPS))+;

            fn context(sources: &Sources, package_watts_mean: Option<f32>) -> Self::Ctx<'_> {
                ($($member::context(sources, package_watts_mean),)+)
            }

            fn consume(acc: &mut Self::Acc, channel: &RawChannel, ctx: &Self::Ctx<'_>) {
                $($member::consume(&mut acc.$index, channel, &ctx.$index);)+
            }

            fn finish(acc: Self::Acc, elapsed: Duration, ctx: &Self::Ctx<'_>) -> Self::Value {
                ($($member::finish(acc.$index, elapsed, &ctx.$index),)+)
            }
        }
    };
}

tuple_impls!(A 0);
tuple_impls!(A 0, B 1);
tuple_impls!(A 0, B 1, C 2);
tuple_impls!(A 0, B 1, C 2, D 3);
tuple_impls!(A 0, B 1, C 2, D 3, E 4);
tuple_impls!(A 0, B 1, C 2, D 3, E 4, F 5);
tuple_impls!(A 0, B 1, C 2, D 3, E 4, F 5, G 6);
tuple_impls!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7);
tuple_impls!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8);
tuple_impls!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9);
tuple_impls!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10);
tuple_impls!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11);

impl Reading for () {
    type Value = ();

    fn read(_sources: &mut Sources) {}
}

impl Measured for () {
    type Value = ();
    type Ctx<'a> = ();
    type Acc = ();
    const GROUPS: IoReportGroups = IoReportGroups::empty();

    fn context(
        _sources: &Sources,
        _package_watts_mean: Option<f32>,
    ) {
    }

    fn consume(
        _acc: &mut (),
        _channel: &RawChannel,
        _ctx: &(),
    ) {
    }

    fn finish(
        _acc: (),
        _elapsed: Duration,
        _ctx: &(),
    ) {
    }
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
