use crate::providers::metrics::BatteryMetrics;

#[cfg(target_os = "macos")]
pub(crate) fn read_battery() -> Option<BatteryMetrics> {
    let snapshot = crate::sys::read_battery()?;
    Some(BatteryMetrics {
        present: snapshot.present,
        percent: snapshot.percent,
        charging: snapshot.charging,
        on_ac_power: snapshot.on_ac_power,
    })
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn read_battery() -> Option<BatteryMetrics> {
    None
}
