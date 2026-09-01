mod error;

pub use error::Error;
#[cfg(target_vendor = "apple")]
use keisoku::{PowerMeter, PowerReading, Watts};
use shoji::types::session::chat::ChatReplyPowerStats;

pub enum PowerRecorder {
    #[cfg(target_vendor = "apple")]
    Apple(PowerMeter),
}

impl PowerRecorder {
    #[cfg(target_vendor = "apple")]
    pub fn new() -> Self {
        Self::Apple(PowerMeter::new())
    }

    pub fn begin(&mut self) -> Result<(), Error> {
        match self {
            #[cfg(target_vendor = "apple")]
            Self::Apple(meter) => {
                meter.start()?;
                Ok(())
            },
        }
    }

    pub fn split(&mut self) -> Result<ChatReplyPowerStats, Error> {
        match self {
            #[cfg(target_vendor = "apple")]
            Self::Apple(meter) => Ok(stats(meter.split()?)),
        }
    }

    pub fn finish(&mut self) -> Result<ChatReplyPowerStats, Error> {
        match self {
            #[cfg(target_vendor = "apple")]
            Self::Apple(meter) => Ok(stats(meter.stop()?)),
        }
    }
}

#[cfg(target_vendor = "apple")]
fn stats(reading: PowerReading) -> ChatReplyPowerStats {
    let watts = |value: Option<Watts>| value.map(|watts| watts.value() as f64);
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
