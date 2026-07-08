#![cfg(target_vendor = "apple")]

use std::cell::RefCell;

use keisoku::{Energy, Interval, Power, Select, Session};
use shoji::types::session::chat::ChatReplyPowerStats;

use crate::util::power::PowerRecorder;

type Meter = Interval<Select![Energy, Power]>;

pub struct ApplePowerRecorder {
    meter: RefCell<Meter>,
    session: RefCell<Option<Session<Select![Energy, Power]>>>,
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
        if !self.meter.borrow().is_available() {
            *self.session.borrow_mut() = None;
            return;
        }
        let session = self.meter.borrow_mut().start();
        *self.session.borrow_mut() = Some(session);
    }

    fn finish(&self) -> Option<ChatReplyPowerStats> {
        if !self.meter.borrow().is_available() {
            return None;
        }
        let session = self.session.borrow_mut().take()?;
        let sample = self.meter.borrow_mut().stop(session);
        let energy = sample.get::<Energy>();
        let average_power = sample.get::<Power>();
        let total_watts = average_power.total().value() as f64;
        Some(ChatReplyPowerStats {
            samples_count: 1,
            average_cpu_watts: average_power.cpu.value() as f64,
            average_gpu_watts: average_power.gpu.value() as f64,
            average_ane_watts: average_power.ane.value() as f64,
            average_ram_watts: average_power.ram.value() as f64,
            average_total_watts: total_watts,
            average_package_watts: total_watts,
            max_package_watts: total_watts,
            energy_joules: energy.total().value() as f64,
        })
    }
}
