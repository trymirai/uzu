use std::time::{Duration, Instant};

#[cfg(target_os = "macos")]
use crate::{
    EnergyModelChannel,
    cpu_load::CpuLoad,
    ioreport::IoReport,
    smc::Smc,
    soc::SocInfo,
    units::{GigabytesPerSecond, Joules, Megahertz, Percent, Watts},
};
#[cfg(target_vendor = "apple")]
use crate::client::SensorReader;
use crate::{
    Component, Device, EnergyReading, EnergyWindow,
    metrics::{
        BandwidthMetrics, CpuMetrics, FanMetrics, GpuMetrics, NeuralEngineMetrics, PowerMetrics, Temperatures,
        read_battery, read_memory, read_thermal_pressure,
    },
    sensor::{Sensor, SensorKind},
    snapshot::Snapshot,
    units::{Celsius, Milliseconds},
};

#[derive(Default)]
struct SocMetrics {
    cpu: Option<CpuMetrics>,
    gpu: Option<GpuMetrics>,
    neural_engine: Option<NeuralEngineMetrics>,
    power: Option<PowerMetrics>,
    bandwidth: Option<BandwidthMetrics>,
}

pub struct Collector {
    #[cfg(target_os = "macos")]
    soc: Option<SocInfo>,
    #[cfg(target_os = "macos")]
    ioreport: Option<IoReport>,
    #[cfg(target_os = "macos")]
    cpu_load: CpuLoad,
    #[cfg(target_os = "macos")]
    smc: Option<Smc>,
    #[cfg(target_vendor = "apple")]
    temperature_reader: Option<SensorReader>,
    #[cfg(target_vendor = "apple")]
    voltage_reader: Option<SensorReader>,
    #[cfg(target_vendor = "apple")]
    current_reader: Option<SensorReader>,
}

impl Default for Collector {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector {
    pub fn new() -> Self {
        Self {
            #[cfg(target_os = "macos")]
            soc: SocInfo::new(),
            #[cfg(target_os = "macos")]
            ioreport: IoReport::new(),
            #[cfg(target_os = "macos")]
            cpu_load: CpuLoad::new(),
            #[cfg(target_os = "macos")]
            smc: Smc::new(),
            #[cfg(target_vendor = "apple")]
            temperature_reader: SensorReader::new(SensorKind::Temperature),
            #[cfg(target_vendor = "apple")]
            voltage_reader: SensorReader::new(SensorKind::Voltage),
            #[cfg(target_vendor = "apple")]
            current_reader: SensorReader::new(SensorKind::Current),
        }
    }

    #[cfg(target_os = "macos")]
    pub fn soc(&self) -> Option<&SocInfo> {
        self.soc.as_ref()
    }

    #[cfg(target_os = "macos")]
    pub fn energy_model_channels(&self) -> Vec<EnergyModelChannel> {
        self.ioreport.as_ref().map(IoReport::energy_model_channels).unwrap_or_default()
    }

    pub fn device(&self) -> Device {
        Device::detect(self)
    }

    #[cfg(target_os = "macos")]
    pub fn start_energy_window(&self) -> Option<EnergyWindow> {
        let sample = self.ioreport.as_ref()?.snapshot()?;
        let package_watts_start = self.smc.as_ref().and_then(|smc| smc.package_watts()).map(|watts| watts.value());
        Some(EnergyWindow::new(sample, Instant::now(), package_watts_start))
    }

    #[cfg(not(target_os = "macos"))]
    pub fn start_energy_window(&self) -> Option<EnergyWindow> {
        None
    }

    #[cfg(target_os = "macos")]
    pub fn end_energy_window(
        &self,
        window: EnergyWindow,
    ) -> Option<EnergyReading> {
        let next = self.ioreport.as_ref()?.snapshot()?;
        let totals = self.ioreport.as_ref()?.energy_delta(&window.sample, &next)?;
        let package_watts_end = self.smc.as_ref().and_then(|smc| smc.package_watts()).map(|watts| watts.value());
        let elapsed = window.started_at.elapsed();
        let mean_package_watts = match (window.package_watts_start, package_watts_end) {
            (Some(start), Some(end)) => Some((start + end) / 2.0),
            _ => None,
        };
        let package_from_smc = mean_package_watts.is_some();
        let elapsed_secs = elapsed.as_secs_f32().max(0.001);
        let package_energy = mean_package_watts.map(|watts| Joules(watts * elapsed_secs)).unwrap_or_else(|| Joules(totals.total()));
        let package_power = mean_package_watts.map(Watts).unwrap_or_else(|| Watts(totals.total() / elapsed_secs));
        Some(EnergyReading {
            energy: totals.energy_metrics(package_energy),
            average_power: totals.power_metrics(elapsed, package_power),
            elapsed: Milliseconds(elapsed.as_millis() as u64),
            package_from_smc,
        })
    }

    #[cfg(not(target_os = "macos"))]
    pub fn end_energy_window(
        &self,
        _window: EnergyWindow,
    ) -> Option<EnergyReading> {
        None
    }

    pub fn sample(
        &mut self,
        interval: Duration,
    ) -> Snapshot {
        let soc = self.sample_soc(interval);
        let memory = read_memory();
        let fans = self.read_fans();
        let battery = read_battery();
        let thermal_pressure = read_thermal_pressure();
        #[cfg(target_vendor = "apple")]
        let sensors = self.temperature_reader.as_mut().map(SensorReader::read).unwrap_or_default();
        #[cfg(not(target_vendor = "apple"))]
        let sensors = crate::sensors(SensorKind::Temperature);
        #[cfg(target_vendor = "apple")]
        let voltage = self.voltage_reader.as_mut().map(SensorReader::read).unwrap_or_default();
        #[cfg(not(target_vendor = "apple"))]
        let voltage = crate::sensors(SensorKind::Voltage);
        #[cfg(target_vendor = "apple")]
        let current = self.current_reader.as_mut().map(SensorReader::read).unwrap_or_default();
        #[cfg(not(target_vendor = "apple"))]
        let current = crate::sensors(SensorKind::Current);
        let temperatures = (!sensors.is_empty()).then(|| temperatures_from(&sensors));
        Snapshot {
            elapsed: Milliseconds(interval.as_millis() as u64),
            cpu: soc.cpu,
            gpu: soc.gpu,
            neural_engine: soc.neural_engine,
            power: soc.power,
            memory,
            bandwidth: soc.bandwidth,
            fans,
            battery,
            temperatures,
            thermal_pressure,
            sensors,
            voltage,
            current,
        }
    }

    #[cfg(target_os = "macos")]
    fn sample_soc(
        &mut self,
        interval: Duration,
    ) -> SocMetrics {
        let (Some(ioreport), Some(soc)) = (&self.ioreport, &self.soc) else {
            std::thread::sleep(interval);
            return SocMetrics::default();
        };
        let sample = ioreport.sample(soc, interval);
        let per_core = self.cpu_load.sample();
        let package = self.smc.as_ref().and_then(|smc| smc.package_watts()).unwrap_or(Watts(sample.total_power));
        SocMetrics {
            cpu: Some(CpuMetrics {
                usage: Percent(sample.cpu_usage_percent * 100.0),
                ecpu_frequency: Megahertz(sample.ecpu_usage.0),
                ecpu_usage: Percent(sample.ecpu_usage.1 * 100.0),
                pcpu_frequency: Megahertz(sample.pcpu_usage.0),
                pcpu_usage: Percent(sample.pcpu_usage.1 * 100.0),
                per_core,
            }),
            gpu: Some(GpuMetrics {
                frequency: Megahertz(sample.gpu_usage.0),
                usage: Percent(sample.gpu_usage.1 * 100.0),
            }),
            neural_engine: Some(NeuralEngineMetrics {
                power: Watts(sample.ane_power),
                active: Percent(sample.ane_active_percent),
                read_bandwidth: GigabytesPerSecond(sample.ane_read_gbps),
                write_bandwidth: GigabytesPerSecond(sample.ane_write_gbps),
            }),
            power: Some(PowerMetrics {
                cpu: Watts(sample.cpu_power),
                gpu: Watts(sample.gpu_power),
                gpu_sram: Watts(sample.gpu_ram_power),
                ane: Watts(sample.ane_power),
                ram: Watts(sample.ram_power),
                total: Watts(sample.total_power),
                package,
            }),
            bandwidth: Some(BandwidthMetrics {
                dram_read: GigabytesPerSecond(sample.dram_read_gbps),
                dram_write: GigabytesPerSecond(sample.dram_write_gbps),
            }),
        }
    }

    #[cfg(not(target_os = "macos"))]
    fn sample_soc(
        &mut self,
        interval: Duration,
    ) -> SocMetrics {
        std::thread::sleep(interval);
        SocMetrics::default()
    }

    #[cfg(target_os = "macos")]
    fn read_fans(&self) -> Option<FanMetrics> {
        self.smc.as_ref().map(|smc| smc.fans())
    }

    #[cfg(not(target_os = "macos"))]
    fn read_fans(&self) -> Option<FanMetrics> {
        None
    }
}

fn average(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
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
