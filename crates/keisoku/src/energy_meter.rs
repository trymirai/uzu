use crate::EnergyReading;

#[cfg(target_os = "macos")]
use std::time::Instant;
#[cfg(target_os = "macos")]
use crate::{
    ioreport::{EnergyTotals, IoReport},
    smc::Smc,
    units::{Joules, Milliseconds, Watts},
};

pub struct EnergyMeter {
    #[cfg(target_os = "macos")]
    io_report: Option<IoReport>,
    #[cfg(target_os = "macos")]
    smc: Option<Smc>,
}

#[cfg(target_os = "macos")]
unsafe impl Send for EnergyMeter {}

impl Default for EnergyMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl EnergyMeter {
    #[cfg(target_os = "macos")]
    pub fn new() -> Self {
        Self {
            io_report: IoReport::energy_only(),
            smc: Smc::new(),
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn new() -> Self {
        Self {}
    }

    #[cfg(target_os = "macos")]
    pub fn start(&self) -> EnergyWindow {
        EnergyWindow {
            start: self.counters(),
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn start(&self) -> EnergyWindow {
        EnergyWindow {}
    }

    #[cfg(target_os = "macos")]
    pub fn stop(
        &self,
        window: EnergyWindow,
    ) -> Option<EnergyReading> {
        let start = window.start?;
        let end = self.counters()?;
        let elapsed = end.at.duration_since(start.at);
        let energy = end.energy.since(&start.energy);
        let mean_package_watts = match (start.package_watts, end.package_watts) {
            (Some(first), Some(last)) => Some((first + last) / 2.0),
            _ => None,
        };
        let package_from_smc = mean_package_watts.is_some();
        let elapsed_secs = elapsed.as_secs_f32().max(0.001);
        let package_energy =
            mean_package_watts.map(|watts| Joules(watts * elapsed_secs)).unwrap_or_else(|| Joules(energy.total() as f32));
        let package_power = mean_package_watts.map(Watts).unwrap_or_else(|| Watts(energy.total() as f32 / elapsed_secs));
        Some(EnergyReading {
            energy: energy.energy_metrics(package_energy),
            average_power: energy.power_metrics(elapsed, package_power),
            elapsed: Milliseconds(elapsed.as_millis() as u64),
            package_from_smc,
        })
    }

    #[cfg(not(target_os = "macos"))]
    pub fn stop(
        &self,
        _window: EnergyWindow,
    ) -> Option<EnergyReading> {
        None
    }

    #[cfg(target_os = "macos")]
    fn counters(&self) -> Option<Counters> {
        let energy = self.io_report.as_ref()?.cumulative_energy()?;
        let package_watts = self.smc.as_ref().and_then(|smc| smc.package_watts()).map(|watts| watts.value());
        Some(Counters {
            energy,
            package_watts,
            at: Instant::now(),
        })
    }
}

#[must_use]
pub struct EnergyWindow {
    #[cfg(target_os = "macos")]
    start: Option<Counters>,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct Counters {
    energy: EnergyTotals,
    package_watts: Option<f32>,
    at: Instant,
}
