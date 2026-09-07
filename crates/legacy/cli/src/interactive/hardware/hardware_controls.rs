use std::{
    fs::{File, OpenOptions},
    io::Write,
};

use keisoku::{FanControl, PowerModeControl};

use super::{HardwareError, PerformanceMode};

/// Owns the process-wide overrides for one interactive session.
#[derive(Default)]
pub struct HardwareControls {
    power_mode: Option<PowerModeControl>,
    fans: Option<FanControl>,
    ownership: Option<File>,
    mode: PerformanceMode,
    closed: bool,
    uncertain: bool,
}

impl HardwareControls {
    pub fn apply(
        &mut self,
        mode: PerformanceMode,
    ) -> Result<(), HardwareError> {
        if self.closed {
            return Err(HardwareError::Closed);
        }
        if !self.uncertain && self.mode == mode {
            return Ok(());
        }
        if self.ownership.is_none() {
            // /var/run is root-owned, so another user cannot substitute the lock file.
            let file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(false)
                .open("/var/run/com.trymirai.cli.hardware.lock")?;
            file.try_lock()?;
            self.ownership = Some(file);
        }
        let previous_mode = self.mode;
        if let Err(operation) = self.apply_inner(mode) {
            return match self.apply_inner(previous_mode) {
                Ok(()) => Err(operation),
                Err(restore) => {
                    self.uncertain = true;
                    Err(HardwareError::Rollback {
                        operation: Box::new(operation),
                        restore: Box::new(restore),
                    })
                },
            };
        }
        Ok(())
    }

    fn apply_inner(
        &mut self,
        mode: PerformanceMode,
    ) -> Result<(), HardwareError> {
        match mode {
            PerformanceMode::Fast => {
                // Open both controllers before mutating either setting.
                if self.power_mode.is_none() {
                    self.power_mode = Some(PowerModeControl::new()?);
                }
                if self.fans.is_none() {
                    self.fans = Some(FanControl::new()?);
                }
                if let Some(fans) = &mut self.fans {
                    fans.set_maximum()?;
                }
                if let Some(power_mode) = &mut self.power_mode {
                    power_mode.set_high()?;
                }
            },
            PerformanceMode::Auto => {
                let power_mode =
                    self.power_mode.as_mut().map_or(Ok(()), PowerModeControl::restore).map_err(HardwareError::from);
                let fans = self.fans.as_mut().map_or(Ok(()), FanControl::restore).map_err(HardwareError::from);
                match (power_mode, fans) {
                    (Ok(()), Ok(())) => {},
                    (Err(operation), Err(restore)) => {
                        return Err(HardwareError::Rollback {
                            operation: Box::new(operation),
                            restore: Box::new(restore),
                        });
                    },
                    (Err(error), _) | (_, Err(error)) => return Err(error),
                }
            },
        }
        self.mode = mode;
        self.uncertain = false;
        if mode == PerformanceMode::Auto {
            self.ownership = None;
        }
        Ok(())
    }

    pub fn restore(&mut self) -> Result<(), HardwareError> {
        self.closed = true;
        self.apply_inner(PerformanceMode::Auto)
    }
}

impl Drop for HardwareControls {
    fn drop(&mut self) {
        if let Err(error) = self.restore() {
            let _ = writeln!(std::io::stderr(), "Unable to restore hardware settings: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancelled_session_rejects_queued_apply_without_touching_hardware() {
        let mut controls = HardwareControls::default();
        controls.restore().unwrap();
        assert!(matches!(controls.apply(PerformanceMode::Auto), Err(HardwareError::Closed)));
    }

    #[test]
    fn default_settings_do_not_acquire_hardware() {
        let mut controls = HardwareControls::default();
        controls.apply(PerformanceMode::Auto).unwrap();
        assert!(controls.ownership.is_none() && controls.power_mode.is_none() && controls.fans.is_none());
    }
}
