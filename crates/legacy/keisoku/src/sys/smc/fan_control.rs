use std::{
    io::Write,
    thread,
    time::{Duration, Instant},
};

use obfstr::obfstr;

use super::{FanState, Smc, SmcError, SmcKeyData};

/// Session-scoped maximum fans. Explicit restoration reports errors; Drop retries best effort.
/// Abrupt termination such as SIGKILL cannot restore hardware state.
pub struct FanControl {
    smc: Smc,
    saved: Vec<FanState>,
    unlock: Option<SmcKeyData>,
    active: bool,
}

impl FanControl {
    /// Opens the SMC connection without changing hardware settings.
    pub fn new() -> Result<Self, SmcError> {
        Ok(Self {
            smc: Smc::connect()?,
            saved: Vec::new(),
            unlock: None,
            active: false,
        })
    }

    /// Saves current fan state and requests each fan's hardware-reported maximum RPM.
    pub fn set_maximum(&mut self) -> Result<(), SmcError> {
        if unsafe { libc::geteuid() } != 0 {
            return Err(SmcError::RootRequired);
        }
        if !self.active {
            self.capture()?;
        }
        if let Err(operation) = self.apply_maximum() {
            return match self.restore() {
                Ok(()) => Err(operation),
                Err(restore) => Err(SmcError::RollbackFailed {
                    operation: Box::new(operation),
                    restore: Box::new(restore),
                }),
            };
        }
        Ok(())
    }

    fn capture(&mut self) -> Result<(), SmcError> {
        let count = self.smc.read_key(obfstr!("FNum"))?.as_u8()?;
        if count == 0 {
            return Err(SmcError::NoFans);
        }
        if count > 10 {
            return Err(SmcError::InvalidData {
                key: 0,
                reason: "unsupported fan count",
            });
        }
        let fans = (0..count).map(|index| FanState::read(&self.smc, index)).collect::<Result<_, _>>()?;
        let unlock = match self.smc.read_key(obfstr!("Ftst")) {
            Ok(value) => {
                if value.as_u8()? > 1 {
                    return Err(SmcError::InvalidData {
                        key: value.key,
                        reason: "unsupported fan unlock state",
                    });
                }
                Some(value)
            },
            Err(SmcError::MissingKey(_)) => None,
            Err(error) => return Err(error),
        };
        self.saved = fans;
        self.unlock = unlock;
        self.active = true;
        Ok(())
    }

    fn apply_maximum(&self) -> Result<(), SmcError> {
        if let Some(unlock) = &self.unlock {
            self.smc.write_checked(unlock, &[1])?;
            thread::sleep(Duration::from_millis(500));
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        for fan in &self.saved {
            loop {
                match self.smc.write_checked(&fan.mode, &[1]) {
                    Ok(()) => break,
                    Err(
                        SmcError::Firmware {
                            ..
                        }
                        | SmcError::ReadbackMismatch(_),
                    ) if Instant::now() < deadline => {
                        thread::sleep(Duration::from_millis(100));
                    },
                    Err(error) => return Err(error),
                }
            }
            self.smc.write_checked(&fan.target, &fan.maximum.to_le_bytes())?;
        }
        Ok(())
    }

    /// Restores the exact saved targets, modes and unlock state; failures remain pending.
    pub fn restore(&mut self) -> Result<(), SmcError> {
        if !self.active {
            return Ok(());
        }
        let mut failure = None;
        for value in self.saved.iter().flat_map(|fan| [&fan.target, &fan.mode]).chain(self.unlock.as_ref()) {
            if let Err(error) = self.smc.write_checked(value, &value.bytes[..value.key_info.data_size as usize]) {
                failure.get_or_insert(error);
            }
        }
        failure.map_or(Ok(()), Err)?;
        self.active = false;
        self.saved.clear();
        self.unlock = None;
        Ok(())
    }
}

impl Drop for FanControl {
    fn drop(&mut self) {
        if let Err(error) = self.restore() {
            let _ = writeln!(std::io::stderr(), "Failed to restore fan control: {error}");
        }
    }
}
