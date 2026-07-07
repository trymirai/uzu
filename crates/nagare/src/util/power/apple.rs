#![cfg(target_vendor = "apple")]

use std::cell::RefCell;

use keisoku::{Energy, Interval, Power, Session};
use shoji::types::session::chat::ChatReplyPowerStats;

use crate::util::power::PowerRecorder;

type Meter = Interval<(Energy, Power)>;

pub struct ApplePowerRecorder {
    meter: RefCell<Meter>,
    session: RefCell<Option<Session<(Energy, Power)>>>,
}

impl ApplePowerRecorder {
    pub fn new() -> Self {
        Self {
            meter: RefCell::new(Meter::new()),
            session: RefCell::new(None),
        }
    }
}

impl PowerRecorder for ApplePowerRecorder {
    fn begin(&self) {
        let session = self.meter.borrow_mut().begin();
        *self.session.borrow_mut() = Some(session);
    }

    fn finish(&self) -> Option<ChatReplyPowerStats> {
        let session = self.session.borrow_mut().take()?;
        let (energy, average_power) = self.meter.borrow_mut().end(session);
        let package_watts = average_power.package.value() as f64;
        Some(ChatReplyPowerStats {
            samples_count: 1,
            average_cpu_watts: average_power.cpu.value() as f64,
            average_gpu_watts: average_power.gpu.value() as f64,
            average_ane_watts: average_power.ane.value() as f64,
            average_ram_watts: average_power.ram.value() as f64,
            average_total_watts: average_power.total().value() as f64,
            average_package_watts: package_watts,
            max_package_watts: package_watts,
            energy_joules: energy.package.value() as f64,
        })
    }
}
