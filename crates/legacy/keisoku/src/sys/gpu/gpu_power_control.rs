use obfstr::obfstr;
use objc2_core_foundation::{CFNumber, CFString};
use objc2_io_kit::{IOObjectRelease, IOObjectRetain, IORegistryEntryCreateCFProperty, io_registry_entry_t};

use super::GpuControlError;
use crate::sys::registry::IoServiceIterator;

/// Read-only access to the GPU's calibrated maximum power budget.
pub struct GpuPowerControl {
    service: io_registry_entry_t,
}

impl GpuPowerControl {
    /// Opens the GPU service without changing its policy.
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
        };
        control.maximum_power_milliwatts()?;
        Ok(control)
    }

    /// Reads the driver's calibrated power budget at its highest performance state.
    /// This is neither an enforced ceiling nor a guarantee of sustainable power or frequency.
    pub fn maximum_power_milliwatts(&self) -> Result<u32, GpuControlError> {
        self.read_power_property(obfstr!("GetGPUMaxPower"))
            .and_then(|value| u32::try_from(value).ok())
            .filter(|&value| value > 0)
            .ok_or(GpuControlError::MaximumPowerUnavailable)
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
}

impl Drop for GpuPowerControl {
    fn drop(&mut self) {
        IOObjectRelease(self.service);
    }
}
