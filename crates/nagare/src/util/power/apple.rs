#![cfg(target_vendor = "apple")]

use std::sync::Mutex;

use shoji::types::session::chat::ChatReplyPowerStats;

use crate::util::power::PowerRecorder;

pub struct ApplePowerRecorder {
    // Option so stop() can take ownership through a shared reference.
    meter: Mutex<Option<keisoku::EnergyMeter>>,
}

impl ApplePowerRecorder {
    pub fn new() -> Self {
        Self {
            meter: Mutex::new(Some(keisoku::EnergyMeter::start())),
        }
    }
}

impl PowerRecorder for ApplePowerRecorder {
    fn stop(&self) -> Option<ChatReplyPowerStats> {
        let reading = self.meter.lock().ok()?.take()?.stop()?;
        let package_watts = reading.average_power.package.value() as f64;
        Some(ChatReplyPowerStats {
            samples_count: 1,
            average_cpu_watts: reading.average_power.cpu.value() as f64,
            average_gpu_watts: reading.average_power.gpu.value() as f64,
            average_gpu_sram_watts: reading.average_power.gpu_sram.value() as f64,
            average_ane_watts: reading.average_power.ane.value() as f64,
            average_ram_watts: reading.average_power.ram.value() as f64,
            average_total_watts: reading.average_power.total().value() as f64,
            average_package_watts: package_watts,
            // A two-point window yields the mean, not a sampled peak.
            max_package_watts: package_watts,
            energy_joules: reading.energy.package.value() as f64,
        })
    }
}
