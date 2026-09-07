use std::io::Write;

use super::{PowerModeError, read_modes, write_mode};

/// High Power Mode for the machine's AC and, when present, battery profiles.
/// Drop retries restoration; abrupt termination cannot restore persistent preferences.
pub struct PowerModeControl {
    originals: Vec<(bool, i64)>,
}

impl PowerModeControl {
    /// Checks native capabilities and preferences without changing either profile.
    pub fn new() -> Result<Self, PowerModeError> {
        read_modes()?;
        Ok(Self {
            originals: Vec::new(),
        })
    }

    /// Requests the platform's High Power Mode, preserving each original preference.
    pub fn set_high(&mut self) -> Result<(), PowerModeError> {
        if unsafe { libc::geteuid() } != 0 {
            return Err(PowerModeError::RootRequired);
        }
        let current = read_modes()?;
        for (source, mode) in current {
            if mode != 2 && !self.originals.iter().any(|(saved, _)| *saved == source) {
                self.originals.push((source, mode));
            }
        }
        for &(source, _) in &self.originals {
            if let Err(operation) = write_mode(source, 2) {
                return match self.restore() {
                    Ok(()) => Err(operation),
                    Err(restore) => Err(PowerModeError::RollbackFailed {
                        operation: Box::new(operation),
                        restore: Box::new(restore),
                    }),
                };
            }
        }
        Ok(())
    }

    /// Restores all saved profiles; failed profiles remain pending for another attempt.
    pub fn restore(&mut self) -> Result<(), PowerModeError> {
        let mut failure = None;
        self.originals.retain(|&(source, mode)| match write_mode(source, mode) {
            Ok(()) => false,
            Err(error) => {
                failure.get_or_insert(error);
                true
            },
        });
        failure.map_or(Ok(()), Err)
    }
}

impl Drop for PowerModeControl {
    fn drop(&mut self) {
        if let Err(error) = self.restore() {
            let _ = writeln!(std::io::stderr(), "Failed to restore power mode: {error}");
        }
    }
}
