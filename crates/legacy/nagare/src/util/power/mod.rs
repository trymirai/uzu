mod error;

pub use error::Error;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use keisoku::{PowerMeter, PowerReading};
use shoji::types::session::chat::ChatReplyEnergy;

pub enum EnergyRecorder {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    Apple(PowerMeter),
}

impl EnergyRecorder {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn new() -> Self {
        Self::Apple(PowerMeter::new())
    }

    pub fn begin(&mut self) -> Result<(), Error> {
        match self {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            Self::Apple(meter) => {
                meter.start()?;
                Ok(())
            },
        }
    }

    pub fn split(&mut self) -> Result<ChatReplyEnergy, Error> {
        match self {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            Self::Apple(meter) => Ok(energy(meter.split()?)),
        }
    }

    pub fn finish(&mut self) -> Result<ChatReplyEnergy, Error> {
        match self {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            Self::Apple(meter) => Ok(energy(meter.stop()?)),
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
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
