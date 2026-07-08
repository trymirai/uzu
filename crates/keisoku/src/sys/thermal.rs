use core::ffi::c_char;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ThermalPressureLevel {
    Nominal,
    Moderate,
    Heavy,
    Trapping,
    Sleeping,
}

impl ThermalPressureLevel {
    fn from_level(level: u64) -> Option<Self> {
        match level {
            0 => Some(ThermalPressureLevel::Nominal),
            1 => Some(ThermalPressureLevel::Moderate),
            2 => Some(ThermalPressureLevel::Heavy),
            3 => Some(ThermalPressureLevel::Trapping),
            4 => Some(ThermalPressureLevel::Sleeping),
            _ => None,
        }
    }
}

pub(crate) fn read_thermal_pressure() -> Option<ThermalPressureLevel> {
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
    ThermalPressureLevel::from_level(level)
}
