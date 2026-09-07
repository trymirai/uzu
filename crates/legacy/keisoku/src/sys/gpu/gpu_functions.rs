use obfstr::obfstr;
use objc2_core_foundation::{CFBoolean, CFDictionary, CFNumber, CFString, CFType};
use objc2_io_kit::io_registry_entry_t;

use super::GpuControlError;

kanka::ffi_table! {
    struct GpuFunctions from "/System/Library/Frameworks/IOKit.framework/IOKit" {
        set_properties = "IORegistryEntrySetCFProperties":
            unsafe extern "C" fn(io_registry_entry_t, &CFDictionary<CFString, CFType>) -> i32,
    }
}

pub fn set_power_limit(
    service: io_registry_entry_t,
    milliwatts: u32,
) -> Result<(), GpuControlError> {
    let functions = GpuFunctions::get().ok_or(GpuControlError::Unavailable)?;
    let trigger = CFString::from_str(obfstr!("SetMaxGPUAbsolutePower"));
    let target = CFString::from_str(obfstr!("AbsoluteTarget"));
    let value = CFNumber::new_i64(i64::from(milliwatts));
    let properties = CFDictionary::<CFString, CFType>::from_slices(
        &[&trigger, &target],
        &[CFBoolean::new(true).as_ref(), value.as_ref()],
    );
    // The dictionary remains alive for the synchronous driver call.
    let code = unsafe { (functions.set_properties)(service, &properties) };
    if code != 0 {
        return Err(GpuControlError::DriverCall {
            operation: "power limit write",
            code,
        });
    }
    Ok(())
}
