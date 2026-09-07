use core::ptr::NonNull;

use obfstr::obfstr;
use objc2_core_foundation::{CFNumber, CFRetained, CFString, CFType};

use super::PowerModeError;

kanka::ffi_table! {
    struct PowerModeFunctions from "/System/Library/Frameworks/IOKit.framework/IOKit" {
        supported_sources = "IOPSGetSupportedPowerSources":
            unsafe extern "C" fn(*mut u32, *mut bool, *mut bool) -> i32,
        available = "IOPMFeatureIsAvailable":
            unsafe extern "C" fn(&CFString, &CFString) -> bool,
        copy_setting = "IOPMCopyPMSetting":
            unsafe extern "C" fn(&CFString, &CFString, *mut *mut CFType) -> i32,
        set_preference = "IOPMSetPMPreference":
            unsafe extern "C" fn(&CFString, &CFType, &CFString) -> i32,
    }
}

pub fn read_modes() -> Result<Vec<(bool, i64)>, PowerModeError> {
    let functions = PowerModeFunctions::get().ok_or(PowerModeError::Unavailable)?;
    let mut has_battery = false;
    let code = unsafe { (functions.supported_sources)(core::ptr::null_mut(), &mut has_battery, core::ptr::null_mut()) };
    check(code, "power source query")?;
    [false, true]
        .into_iter()
        .take(if has_battery {
            2
        } else {
            1
        })
        .map(|battery| {
            let source = source(battery);
            let feature = CFString::from_str(obfstr!("HighPowerMode"));
            if !unsafe { (functions.available)(&feature, &source) } {
                return Err(PowerModeError::Unsupported(if battery {
                    "battery"
                } else {
                    "AC"
                }));
            }
            Ok((battery, read_mode(functions, &source)?))
        })
        .collect()
}

pub fn write_mode(
    battery: bool,
    mode: i64,
) -> Result<(), PowerModeError> {
    let functions = PowerModeFunctions::get().ok_or(PowerModeError::Unavailable)?;
    let source = source(battery);
    let key = CFString::from_str(obfstr!("LowPowerMode"));
    let number = CFNumber::new_i64(mode);
    let code = unsafe { (functions.set_preference)(&key, number.as_ref(), &source) };
    check(code, "preference write")?;
    let actual = read_mode(functions, &source)?;
    if actual != mode {
        return Err(PowerModeError::ReadbackMismatch {
            requested: mode,
            actual,
        });
    }
    Ok(())
}

fn source(battery: bool) -> CFRetained<CFString> {
    if battery {
        CFString::from_str(obfstr!("Battery Power"))
    } else {
        CFString::from_str(obfstr!("AC Power"))
    }
}

fn read_mode(
    functions: &PowerModeFunctions,
    source: &CFString,
) -> Result<i64, PowerModeError> {
    let key = CFString::from_str(obfstr!("LowPowerMode"));
    let mut raw = core::ptr::null_mut();
    let code = unsafe { (functions.copy_setting)(&key, source, &mut raw) };
    // Copy-rule ownership applies to any returned object, including an error reply.
    let value = NonNull::new(raw).map(|pointer| unsafe { CFRetained::<CFType>::from_raw(pointer) });
    check(code, "preference read")?;
    value
        .and_then(|value| value.downcast::<CFNumber>().ok())
        .and_then(|number| number.as_i64())
        .filter(|mode| (0..=2).contains(mode))
        .ok_or(PowerModeError::InvalidPreference)
}

fn check(
    code: i32,
    operation: &'static str,
) -> Result<(), PowerModeError> {
    if code != 0 {
        return Err(PowerModeError::PlatformCall {
            operation,
            code,
        });
    }
    Ok(())
}
