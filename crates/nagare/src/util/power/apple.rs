#![cfg(target_vendor = "apple")]

use std::cell::RefCell;

use keisoku::{PowerMeter, PowerReading, Watts};
use shoji::types::session::chat::ChatReplyPowerStats;

use crate::util::power::PowerRecorder;

pub struct ApplePowerRecorder {
    meter: RefCell<PowerMeter>,
}

impl ApplePowerRecorder {
    pub fn new() -> Self {
        Self {
            meter: RefCell::new(PowerMeter::new()),
        }
    }
}

impl PowerRecorder for ApplePowerRecorder {
    fn begin(&self) {
        self.meter.borrow_mut().start();
    }

    fn finish(&self) -> Option<ChatReplyPowerStats> {
        self.meter.borrow_mut().stop().map(stats)
    }
}

fn stats(reading: PowerReading) -> ChatReplyPowerStats {
    let watts = |value: Option<Watts>| value.map_or(0.0, |watts| watts.value() as f64);
    ChatReplyPowerStats {
        samples_count: reading.samples as i64,
        average_cpu_watts: watts(reading.cpu),
        average_gpu_watts: watts(reading.gpu),
        average_ane_watts: watts(reading.ane),
        average_ram_watts: watts(reading.ram),
        average_total_watts: reading.total.value() as f64,
        energy_joules: reading.energy.value() as f64,
    }
}
