use serde::{Deserialize, Serialize};

use crate::{
    component::Component,
    metrics::{BatteryMetrics, FanMetrics, MemoryMetrics, Temperatures, ThermalPressure},
    sensor::Sensor,
    units::Watts,
};

/// Instantaneous telemetry — each field is meaningful from a single read (no window). Sourced from
/// HID / SMC / sysinfo / IOKit, so reading these never builds an IOReport subscription.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Gauges {
    pub memory: Option<MemoryMetrics>,
    pub fans: Option<FanMetrics>,
    pub battery: Option<BatteryMetrics>,
    pub temperatures: Option<Temperatures>,
    pub thermal_pressure: Option<ThermalPressure>,
    pub package_watts: Option<Watts>,
    pub sensors: Box<[Sensor]>,
    #[serde(default)]
    pub voltage: Box<[Sensor]>,
    #[serde(default)]
    pub current: Box<[Sensor]>,
}

impl Gauges {
    /// Sum of charger/battery watts from matched HID voltage×current rails
    /// (e.g. `Charger VQ0u`×`Charger IQ0u`). An estimate, unlike IOReport per-rail power.
    pub fn rail_power(&self) -> Option<Watts> {
        // Power magnitude so battery discharge (negative current when unplugged) counts too;
        // the ceiling rejects mismatched-unit pairs, well above real charge/discharge.
        const MAX_PLAUSIBLE_WATTS: f64 = 1000.0;
        let split_area_code = |name: &str| name.rsplit_once(' ').map(|(area, code)| (area.to_owned(), code.to_owned()));
        let is_battery_rail = |sensor: &&Sensor| matches!(sensor.component, Component::Charger | Component::Battery);

        let mut total_watts = 0f64;
        for voltage_sensor in self.voltage.iter().filter(is_battery_rail) {
            let Some((voltage_area, voltage_code)) = split_area_code(&voltage_sensor.name) else {
                continue;
            };
            let Some(rail_code) = voltage_code.strip_prefix('V').filter(|code| !code.is_empty()) else {
                continue;
            };
            for current_sensor in self.current.iter().filter(is_battery_rail) {
                let Some((current_area, current_code)) = split_area_code(&current_sensor.name) else {
                    continue;
                };
                if current_area == voltage_area && current_code.strip_prefix('I') == Some(rail_code) {
                    let watts = (voltage_sensor.value * current_sensor.value).abs();
                    if (0.0..=MAX_PLAUSIBLE_WATTS).contains(&watts) {
                        total_watts += watts;
                    }
                }
            }
        }
        (total_watts > 0.0).then_some(Watts(total_watts as f32))
    }
}
