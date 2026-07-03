use std::time::Duration;

#[cfg(target_os = "macos")]
use crate::{
    EnergyModelChannel,
    cpu_load::CpuLoad,
    ioreport::IoReport,
    smc::Smc,
    soc::SocInfo,
    units::{GigabytesPerSecond, Megahertz, Percent, Watts},
};
use crate::{
    Component, Device, Gauges,
    client::SensorReader,
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
    temperature_reader: Option<SensorReader>,
    voltage_reader: Option<SensorReader>,
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
            temperature_reader: SensorReader::new(SensorKind::Temperature),
            voltage_reader: SensorReader::new(SensorKind::Voltage),
            current_reader: SensorReader::new(SensorKind::Current),
        }
    }

    #[cfg(target_os = "macos")]
    pub fn soc(&self) -> Option<&SocInfo> {
        self.soc.as_ref()
    }

    #[cfg(target_os = "macos")]
    pub fn energy_model_channels(&self) -> Box<[EnergyModelChannel]> {
        self.ioreport.as_ref().map(IoReport::energy_model_channels).unwrap_or_default()
    }

    pub fn device(&self) -> Device {
        Device::detect(self)
    }

    /// Instantaneous telemetry only (RAM, temps, HID sensors, fans, battery, thermal pressure, SMC
    /// package watts). No IOReport subscription, so this is cheap — unlike [`Collector::sample`].
    pub fn gauges(&mut self) -> Gauges {
        let sensors = self.temperature_reader.as_mut().map(SensorReader::read).unwrap_or_default();
        let voltage = self.voltage_reader.as_mut().map(SensorReader::read).unwrap_or_default();
        let current = self.current_reader.as_mut().map(SensorReader::read).unwrap_or_default();
        let temperatures = (!sensors.is_empty()).then(|| temperatures_from(&sensors));
        #[cfg(target_os = "macos")]
        let package_watts = self.smc.as_ref().and_then(|smc| smc.package_watts());
        #[cfg(not(target_os = "macos"))]
        let package_watts = None;
        Gauges {
            memory: read_memory(),
            fans: self.read_fans(),
            battery: read_battery(),
            temperatures,
            thermal_pressure: read_thermal_pressure(),
            package_watts,
            sensors,
            voltage,
            current,
        }
    }

    pub fn sample(
        &mut self,
        interval: Duration,
    ) -> Snapshot {
        let soc = self.sample_soc(interval);
        let gauges = self.gauges();
        Snapshot {
            elapsed: Milliseconds(interval.as_millis() as u64),
            cpu: soc.cpu,
            gpu: soc.gpu,
            neural_engine: soc.neural_engine,
            power: soc.power,
            bandwidth: soc.bandwidth,
            memory: gauges.memory,
            fans: gauges.fans,
            battery: gauges.battery,
            temperatures: gauges.temperatures,
            thermal_pressure: gauges.thermal_pressure,
            sensors: gauges.sensors,
            voltage: gauges.voltage,
            current: gauges.current,
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
                active: Percent(sample.ane_active_percent),
            }),
            power: Some(PowerMetrics {
                cpu: Watts(sample.cpu_power),
                gpu: Watts(sample.gpu_power),
                gpu_sram: Watts(sample.gpu_ram_power),
                ane: Watts(sample.ane_power),
                ram: Watts(sample.ram_power),
                package,
            }),
            bandwidth: Some(BandwidthMetrics {
                dram_read: GigabytesPerSecond(sample.dram_read_gbps),
                dram_write: GigabytesPerSecond(sample.dram_write_gbps),
                ane_read: GigabytesPerSecond(sample.ane_read_gbps),
                ane_write: GigabytesPerSecond(sample.ane_write_gbps),
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
