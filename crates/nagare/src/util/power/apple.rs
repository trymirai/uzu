#![cfg(target_vendor = "apple")]

use std::cell::RefCell;

use shoji::types::session::chat::ChatReplyPowerStats;

use crate::util::power::PowerRecorder;

pub struct ApplePowerRecorder {
    meter: keisoku::EnergyMeter,
    window: RefCell<Option<keisoku::EnergyWindow>>,
}

impl ApplePowerRecorder {
    pub fn new() -> Self {
        Self {
            meter: keisoku::EnergyMeter::new(),
            window: RefCell::new(None),
        }
    }
}

impl PowerRecorder for ApplePowerRecorder {
    fn begin(&self) {
        *self.window.borrow_mut() = Some(self.meter.start());
    }

    fn finish(&self) -> Option<ChatReplyPowerStats> {
        let window = self.window.borrow_mut().take()?;
        let reading = self.meter.stop(window)?;
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
            max_package_watts: package_watts,
            energy_joules: reading.energy.package.value() as f64,
        })
    }
}
