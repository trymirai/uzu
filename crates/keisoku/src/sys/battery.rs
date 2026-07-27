use objc2_core_foundation::{CFBoolean, CFNumber, CFString, CFType};
use objc2_io_kit::{
    IOPSCopyPowerSourcesInfo, IOPSCopyPowerSourcesList, IOPSGetPowerSourceDescription, kIOPSCurrentCapacityKey,
    kIOPSInternalBatteryType, kIOPSIsChargingKey, kIOPSMaxCapacityKey, kIOPSPowerSourceStateKey, kIOPSTypeKey,
};

use crate::{metrics::BatteryMetrics, sys::registry::dictionary_get, units::Percent};

pub(crate) fn read_battery() -> Option<BatteryMetrics> {
    let info = IOPSCopyPowerSourcesInfo()?;
    let info_ref: &CFType = &info;
    let list = unsafe { IOPSCopyPowerSourcesList(Some(info_ref)) }?;
    let internal = kIOPSInternalBatteryType.to_str().ok()?;

    for index in 0..list.count() {
        let source = unsafe { list.value_at_index(index) };
        let Some(source) = (unsafe { (source as *const CFType).as_ref() }) else {
            continue;
        };
        let Some(description) = (unsafe { IOPSGetPowerSourceDescription(Some(info_ref), Some(source)) }) else {
            continue;
        };
        let string =
            |key: &core::ffi::CStr| dictionary_get::<CFString>(&description, key.to_str().ok()?).map(|s| s.to_string());
        let number = |key: &core::ffi::CStr| {
            dictionary_get::<CFNumber>(&description, key.to_str().ok()?).and_then(|n| n.as_i64())
        };
        let boolean =
            |key: &core::ffi::CStr| dictionary_get::<CFBoolean>(&description, key.to_str().ok()?).map(|b| b.value());

        if string(kIOPSTypeKey)? != internal {
            continue;
        }
        let current = number(kIOPSCurrentCapacityKey)?;
        let maximum = number(kIOPSMaxCapacityKey)?;
        let charging = boolean(kIOPSIsChargingKey).unwrap_or(false);
        let on_ac_power = string(kIOPSPowerSourceStateKey).is_some_and(|state| state.eq_ignore_ascii_case("AC Power"));
        let percent = if maximum > 0 {
            (current * 100 / maximum) as f32
        } else {
            0.0
        };

        return Some(BatteryMetrics {
            present: true,
            percent: Percent(percent),
            charging,
            on_ac_power,
        });
    }
    None
}
