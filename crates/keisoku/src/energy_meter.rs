use crate::EnergyReading;

#[cfg(target_os = "macos")]
use std::time::Instant;
#[cfg(target_os = "macos")]
use crate::{
    ioreport::{EnergyTotals, IoReport},
    smc::Smc,
    units::{Joules, Milliseconds, Watts},
};

#[must_use]
pub struct EnergyMeter {
    #[cfg(target_os = "macos")]
    start: Option<Counters>,
}

impl EnergyMeter {
    #[cfg(target_os = "macos")]
    pub fn start() -> Self {
        Self {
            start: Counters::read(),
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn start() -> Self {
        Self {}
    }

    #[cfg(target_os = "macos")]
    pub fn stop(self) -> Option<EnergyReading> {
        let start = self.start?;
        let end = Counters::read()?;
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
    pub fn stop(self) -> Option<EnergyReading> {
        None
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct Counters {
    energy: EnergyTotals,
    package_watts: Option<f32>,
    at: Instant,
}

#[cfg(target_os = "macos")]
impl Counters {
    fn read() -> Option<Self> {
        let energy = IoReport::energy_only()?.cumulative_energy()?;
        let package_watts = Smc::new().and_then(|smc| smc.package_watts()).map(|watts| watts.value());
        Some(Self {
            energy,
            package_watts,
            at: Instant::now(),
        })
    }
}
