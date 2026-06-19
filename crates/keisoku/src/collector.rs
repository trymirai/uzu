use std::time::Duration;

#[cfg(target_os = "macos")]
use crate::units::{GigabytesPerSecond, Megahertz, Percent, Watts};
use crate::{
    metrics::{BandwidthMetrics, CpuMetrics, GpuMetrics, NeuralEngineMetrics, PowerMetrics, Temperatures},
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
    soc: Option<crate::soc::SocInfo>,
    #[cfg(target_os = "macos")]
    ioreport: Option<crate::ioreport::IoReport>,
    #[cfg(target_os = "macos")]
    cpu_load: crate::cpu_load::CpuLoad,
    #[cfg(target_os = "macos")]
    smc: Option<crate::smc::Smc>,
    #[cfg(target_vendor = "apple")]
    temperature_reader: Option<crate::client::SensorReader>,
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
            soc: crate::soc::SocInfo::new(),
            #[cfg(target_os = "macos")]
            ioreport: crate::ioreport::IoReport::new(),
            #[cfg(target_os = "macos")]
            cpu_load: crate::cpu_load::CpuLoad::new(),
            #[cfg(target_os = "macos")]
            smc: crate::smc::Smc::new(),
            #[cfg(target_vendor = "apple")]
            temperature_reader: crate::client::SensorReader::new(SensorKind::Temperature),
        }
    }

    #[cfg(target_os = "macos")]
    pub fn soc(&self) -> Option<&crate::soc::SocInfo> {
        self.soc.as_ref()
    }

    pub fn device(&self) -> crate::Device {
        crate::Device::detect(self)
    }

    pub fn sample(
        &mut self,
        interval: Duration,
    ) -> Snapshot {
        let soc = self.sample_soc(interval);
        let memory = crate::metrics::read_memory();
        let fans = self.read_fans();
        let battery = crate::metrics::read_battery();
        let thermal_pressure = crate::metrics::read_thermal_pressure();
        #[cfg(target_vendor = "apple")]
        let sensors = self.temperature_reader.as_mut().map(crate::client::SensorReader::read).unwrap_or_default();
        #[cfg(not(target_vendor = "apple"))]
        let sensors = crate::sensors(SensorKind::Temperature);
        let temperatures = (!sensors.is_empty()).then(|| temperatures_from(&sensors));
        Snapshot {
            elapsed: Milliseconds(0),
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
    fn read_fans(&self) -> Option<crate::metrics::FanMetrics> {
        self.smc.as_ref().map(|smc| smc.fans())
    }

    #[cfg(not(target_os = "macos"))]
    fn read_fans(&self) -> Option<crate::metrics::FanMetrics> {
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
    use crate::Component;
    let average_of = |components: &[Component]| {
        let values: Vec<f32> = sensors
            .iter()
            .filter(|sensor| components.contains(&sensor.component) && (1.0..150.0).contains(&sensor.value))
            .map(|sensor| sensor.value as f32)
            .collect();
        Celsius(average(&values))
    };
    Temperatures {
        cpu_average: average_of(&[Component::Cpu, Component::Soc]),
        gpu_average: average_of(&[Component::Gpu]),
    }
}
