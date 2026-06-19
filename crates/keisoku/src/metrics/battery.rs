use serde::{Deserialize, Serialize};

use crate::units::Percent;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct BatteryMetrics {
    pub present: bool,
    pub percent: Percent,
    pub charging: bool,
    pub on_ac_power: bool,
}

#[cfg(target_os = "macos")]
pub(crate) fn read() -> Option<BatteryMetrics> {
    use objc2_core_foundation::{CFBoolean, CFNumber, CFType};
    use objc2_io_kit::{
        IOPSCopyPowerSourcesInfo, IOPSCopyPowerSourcesList, IOPSGetPowerSourceDescription, kIOPSCurrentCapacityKey,
        kIOPSInternalBatteryType, kIOPSIsChargingKey, kIOPSMaxCapacityKey, kIOPSPowerSourceStateKey, kIOPSTypeKey,
    };

    use crate::cf::{cf_dictionary_value, cf_string_to_string};

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
        let value = |key: &core::ffi::CStr| cf_dictionary_value(&description, key.to_str().ok()?);

        if cf_string_to_string(value(kIOPSTypeKey)?) != internal {
            continue;
        }
        let current = unsafe { &*(value(kIOPSCurrentCapacityKey)? as *const CFNumber) }.as_i64()?;
        let maximum = unsafe { &*(value(kIOPSMaxCapacityKey)? as *const CFNumber) }.as_i64()?;
        let charging =
            value(kIOPSIsChargingKey).map(|ptr| unsafe { &*(ptr as *const CFBoolean) }.value()).unwrap_or(false);
        let on_ac_power = value(kIOPSPowerSourceStateKey)
            .map(cf_string_to_string)
            .is_some_and(|state| state.eq_ignore_ascii_case("AC Power"));
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

#[cfg(not(target_os = "macos"))]
pub(crate) fn read() -> Option<BatteryMetrics> {
    None
}
