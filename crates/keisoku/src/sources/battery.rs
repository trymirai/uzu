use crate::metrics::BatteryMetrics;

#[cfg(target_os = "macos")]
pub(crate) fn read_battery() -> Option<BatteryMetrics> {
    crate::sys::read_battery()
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn read_battery() -> Option<BatteryMetrics> {
    None
}
