mod error;

pub use error::Error;
use keisoku::{PowerMeter, PowerReading};
use shoji::types::session::chat::ChatReplyEnergy;

pub struct EnergyRecorder {
    meter: PowerMeter,
}

impl EnergyRecorder {
    pub fn new() -> Self {
        Self {
            meter: PowerMeter::new(),
        }
    }

    pub fn begin(&mut self) -> Result<(), Error> {
        self.meter.start()?;
        Ok(())
    }

    pub fn split(&mut self) -> Result<ChatReplyEnergy, Error> {
        Ok(energy(self.meter.split()?))
    }

    pub fn finish(&mut self) -> Result<ChatReplyEnergy, Error> {
        Ok(energy(self.meter.stop()?))
    }
}

fn energy(reading: PowerReading) -> ChatReplyEnergy {
    match reading {
        PowerReading::Total {
            total,
        } => ChatReplyEnergy::Total {
            total: total.value() as f64,
        },
        PowerReading::Components {
            cpu,
            gpu,
            ane,
            ram,
        } => ChatReplyEnergy::Components {
            cpu: cpu.value() as f64,
            gpu: gpu.value() as f64,
            ane: ane.value() as f64,
            dram: ram.value() as f64,
        },
    }
}
