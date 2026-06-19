//! Gathers a [`Snapshot`] from whichever providers are available on this
//! platform (IOReport / memory / IOHID sensors / thermal pressure).

use std::time::Duration;

#[cfg(target_os = "macos")]
use crate::units::{GigabytesPerSecond, Megahertz, Percent, Watts};
use crate::{
    metrics::{BandwidthMetrics, CpuMetrics, GpuMetrics, NeuralEngineMetrics, PowerMetrics, Temperatures},
    sensor::{Sensor, SensorKind},
    snapshot::Snapshot,
    units::{Celsius, Milliseconds},
};

/// The SoC metrics IOReport derives in one sample (macOS only).
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
        }
    }

    /// Static SoC description (macOS only).
    #[cfg(target_os = "macos")]
    pub fn soc(&self) -> Option<&crate::soc::SocInfo> {
        self.soc.as_ref()
    }

    /// Collects one snapshot. **Blocks for ~`interval`**: on macOS the IOReport
    /// energy/residency delta needs a real window (which also sets the cadence);
    /// otherwise it just sleeps `interval`.
    pub fn sample(
        &mut self,
        interval: Duration,
    ) -> Snapshot {
        let soc = self.sample_soc(interval);
        let memory = crate::metrics::read_memory();
        let thermal_pressure = crate::metrics::read_thermal_pressure();
        let sensors = crate::sensors(SensorKind::Temperature);
        let temperatures = (!sensors.is_empty()).then(|| temperatures_from(&sensors));
        Snapshot {
            elapsed_milliseconds: Milliseconds(0),
            cpu: soc.cpu,
            gpu: soc.gpu,
            neural_engine: soc.neural_engine,
            power: soc.power,
            memory,
            bandwidth: soc.bandwidth,
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
        SocMetrics {
            cpu: Some(CpuMetrics {
                usage_percent: Percent(sample.cpu_usage_percent * 100.0),
                ecpu_frequency_megahertz: Megahertz(sample.ecpu_usage.0),
                ecpu_usage_percent: Percent(sample.ecpu_usage.1 * 100.0),
                pcpu_frequency_megahertz: Megahertz(sample.pcpu_usage.0),
                pcpu_usage_percent: Percent(sample.pcpu_usage.1 * 100.0),
            }),
            gpu: Some(GpuMetrics {
                frequency_megahertz: Megahertz(sample.gpu_usage.0),
                usage_percent: Percent(sample.gpu_usage.1 * 100.0),
            }),
            neural_engine: Some(NeuralEngineMetrics {
                power_watts: Watts(sample.ane_power),
                active_percent: Percent(sample.ane_active_percent),
                read_bandwidth_gbps: GigabytesPerSecond(sample.ane_read_gbps),
                write_bandwidth_gbps: GigabytesPerSecond(sample.ane_write_gbps),
            }),
            power: Some(PowerMetrics {
                cpu_watts: Watts(sample.cpu_power),
                gpu_watts: Watts(sample.gpu_power),
                gpu_sram_watts: Watts(sample.gpu_ram_power),
                ane_watts: Watts(sample.ane_power),
                ram_watts: Watts(sample.ram_power),
                total_watts: Watts(sample.total_power),
            }),
            bandwidth: Some(BandwidthMetrics {
                dram_read_gbps: GigabytesPerSecond(sample.dram_read_gbps),
                dram_write_gbps: GigabytesPerSecond(sample.dram_write_gbps),
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
}

fn average(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

/// Derives CPU/GPU average temperatures from classified sensors.
fn temperatures_from(sensors: &[Sensor]) -> Temperatures {
    use crate::Component;
    let cpu_temperatures: Vec<f32> = sensors
        .iter()
        .filter(|sensor| matches!(sensor.component, Component::Cpu | Component::Soc))
        .map(|sensor| sensor.value as f32)
        .collect();
    let gpu_temperatures: Vec<f32> =
        sensors.iter().filter(|sensor| sensor.component == Component::Gpu).map(|sensor| sensor.value as f32).collect();
    Temperatures {
        cpu_average: Celsius(average(&cpu_temperatures)),
        gpu_average: Celsius(average(&gpu_temperatures)),
    }
}
