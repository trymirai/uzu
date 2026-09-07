use std::io::Write;

use obfstr::obfstr;
use objc2_core_foundation::{CFNumber, CFString};
use objc2_io_kit::{IOObjectRelease, IOObjectRetain, IORegistryEntryCreateCFProperty, io_registry_entry_t};

use super::{GpuControlError, set_power_limit};
use crate::sys::registry::IoServiceIterator;

/// A GPU power ceiling with restoration of the value preceding the first change.
/// Raising the ceiling does not force power consumption or a particular frequency.
pub struct GpuPowerControl {
    service: io_registry_entry_t,
    original_limit: Option<u32>,
}

impl GpuPowerControl {
    /// Opens a readable GPU service without changing its power limit.
    pub fn new() -> Result<Self, GpuControlError> {
        let mut services = IoServiceIterator::new(obfstr!("AGXAccelerator")).ok_or(GpuControlError::Unavailable)?;
        let (service, _) = services.next().ok_or(GpuControlError::Unavailable)?;
        let code = IOObjectRetain(service);
        if code != 0 {
            return Err(GpuControlError::DriverCall {
                operation: "service retain",
                code,
            });
        }
        let control = Self {
            service,
            original_limit: None,
        };
        control.power_limit_milliwatts()?;
        Ok(control)
    }

    pub fn power_limit_milliwatts(&self) -> Result<u32, GpuControlError> {
        positive_milliwatts(
            self.read_power_property(obfstr!("MaxGPUAbsolutePower")).ok_or(GpuControlError::PowerLimitUnavailable)?,
        )
    }

    /// Reads the driver's calibrated power budget at its highest performance state.
    /// This is neither an enforced ceiling nor a guarantee of sustainable power or frequency.
    pub fn maximum_power_milliwatts(&self) -> Result<u32, GpuControlError> {
        positive_milliwatts(
            self.read_power_property(obfstr!("GetGPUMaxPower")).ok_or(GpuControlError::MaximumPowerUnavailable)?,
        )
    }

    fn read_power_property(
        &self,
        name: &str,
    ) -> Option<i64> {
        let key = CFString::from_str(name);
        // This dynamic driver property is omitted from bulk registry dictionaries.
        unsafe { IORegistryEntryCreateCFProperty(self.service, Some(&key), None, 0) }
            .and_then(|value| value.downcast::<CFNumber>().ok())
            .and_then(|number| number.as_i64())
    }

    /// Changes the ceiling and verifies its readback, rolling back a failed change.
    pub fn set_power_limit_milliwatts(
        &mut self,
        milliwatts: u32,
    ) -> Result<(), GpuControlError> {
        positive_milliwatts(i64::from(milliwatts))?;
        let current = self.power_limit_milliwatts()?;
        let mut original = self.original_limit;
        let result = update_limit(&mut original, current, milliwatts, |limit| self.write_limit(limit));
        self.original_limit = original;
        result
    }

    /// Restores the original ceiling, retaining it for retry if verification fails.
    pub fn restore(&mut self) -> Result<(), GpuControlError> {
        if let Some(original) = self.original_limit {
            self.write_limit(original)?;
            self.original_limit = None;
        }
        Ok(())
    }

    fn write_limit(
        &self,
        milliwatts: u32,
    ) -> Result<(), GpuControlError> {
        set_power_limit(self.service, milliwatts)?;
        let actual = self.power_limit_milliwatts()?;
        if actual != milliwatts {
            return Err(GpuControlError::ReadbackMismatch {
                requested: milliwatts,
                actual,
            });
        }
        Ok(())
    }
}

impl Drop for GpuPowerControl {
    fn drop(&mut self) {
        if let Err(error) = self.restore() {
            let _ = writeln!(std::io::stderr(), "Failed to restore GPU power limit: {error}");
        }
        IOObjectRelease(self.service);
    }
}

fn positive_milliwatts(value: i64) -> Result<u32, GpuControlError> {
    let value = u32::try_from(value)?;
    if value == 0 {
        return Err(GpuControlError::InvalidPowerLimit);
    }
    Ok(value)
}

fn update_limit(
    original: &mut Option<u32>,
    current: u32,
    requested: u32,
    mut write: impl FnMut(u32) -> Result<(), GpuControlError>,
) -> Result<(), GpuControlError> {
    if current == requested {
        return Ok(());
    }
    let previous_original = *original;
    original.get_or_insert(current);
    if let Err(operation) = write(requested) {
        if let Err(rollback) = write(current) {
            return Err(GpuControlError::RollbackFailed {
                operation: Box::new(operation),
                rollback: Box::new(rollback),
            });
        }
        *original = previous_original;
        return Err(operation);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{GpuControlError, positive_milliwatts, update_limit};

    #[test]
    fn milliwatts_require_positive_representable_values() {
        for value in [i64::MIN, -1, 0, i64::from(u32::MAX) + 1] {
            assert!(positive_milliwatts(value).is_err());
        }
        assert_eq!(positive_milliwatts(1).unwrap(), 1);
        assert_eq!(positive_milliwatts(i64::from(u32::MAX)).unwrap(), u32::MAX);
    }

    #[test]
    fn repeated_changes_preserve_original_and_allow_higher_limits() {
        let mut original = None;
        let mut writes = Vec::new();
        for (current, requested) in [(94_000, 120_000), (120_000, 150_000)] {
            update_limit(&mut original, current, requested, |limit| {
                writes.push(limit);
                Ok(())
            })
            .unwrap();
        }
        assert_eq!(original, Some(94_000));
        assert_eq!(writes, [120_000, 150_000]);
    }

    #[test]
    fn failed_change_restores_current_without_discarding_earlier_original() {
        for prior in [None, Some(94_000)] {
            let mut original = prior;
            let mut writes = Vec::new();
            let result = update_limit(&mut original, 120_000, 150_000, |limit| {
                writes.push(limit);
                if limit == 150_000 {
                    Err(GpuControlError::ReadbackMismatch {
                        requested: limit,
                        actual: 140_000,
                    })
                } else {
                    Ok(())
                }
            });
            assert!(matches!(result, Err(GpuControlError::ReadbackMismatch { .. })));
            assert_eq!(writes, [150_000, 120_000]);
            assert_eq!(original, prior);
        }
    }

    #[test]
    fn failed_rollback_preserves_original_for_retry() {
        let mut original = None;
        let mut writes = Vec::new();
        let result = update_limit(&mut original, 94_000, 150_000, |limit| {
            writes.push(limit);
            Err(GpuControlError::PowerLimitUnavailable)
        });
        assert!(matches!(result, Err(GpuControlError::RollbackFailed { .. })));
        assert_eq!(writes, [150_000, 94_000]);
        assert_eq!(original, Some(94_000));
    }

    #[test]
    fn unchanged_limit_does_not_write_or_capture_baseline() {
        let mut original = None;
        update_limit(&mut original, 94_000, 94_000, |_| panic!("unexpected write")).unwrap();
        assert_eq!(original, None);
    }
}
