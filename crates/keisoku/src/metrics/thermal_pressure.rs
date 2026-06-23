use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermalPressure {
    Nominal,
    Moderate,
    Heavy,
    Trapping,
    Sleeping,
}

#[cfg(target_vendor = "apple")]
impl ThermalPressure {
    fn from_level(level: u64) -> Option<Self> {
        match level {
            0 => Some(ThermalPressure::Nominal),
            1 => Some(ThermalPressure::Moderate),
            2 => Some(ThermalPressure::Heavy),
            3 => Some(ThermalPressure::Trapping),
            4 => Some(ThermalPressure::Sleeping),
            _ => None,
        }
    }
}

#[cfg(target_vendor = "apple")]
pub(crate) fn read() -> Option<ThermalPressure> {
    use core::ffi::c_char;

    unsafe extern "C" {
        fn notify_register_check(
            name: *const c_char,
            out_token: *mut i32,
        ) -> u32;
        fn notify_get_state(
            token: i32,
            out_state: *mut u64,
        ) -> u32;
        fn notify_cancel(token: i32) -> u32;
    }

    let name = std::ffi::CString::new("com.apple.system.thermalpressurelevel").ok()?;
    let mut token = 0i32;
    if unsafe { notify_register_check(name.as_ptr(), &mut token) } != 0 {
        return None;
    }
    let mut level = 0u64;
    let result = unsafe { notify_get_state(token, &mut level) };
    unsafe { notify_cancel(token) };
    if result != 0 {
        return None;
    }
    ThermalPressure::from_level(level)
}

#[cfg(not(target_vendor = "apple"))]
pub(crate) fn read() -> Option<ThermalPressure> {
    None
}
