use std::time::Duration;

use crate::{component::Component, sensor::Sensor, units::Joules};

#[derive(Clone, Copy)]
enum PowerFlow {
    Charging,
    Discharging,
}

impl PowerFlow {
    fn parse_voltage_code(voltage_code: &str) -> Option<(Self, &str)> {
        let (flow, rail_code) = if let Some(rail_code) = voltage_code.strip_prefix('V') {
            (Self::Charging, rail_code)
        } else {
            (Self::Discharging, voltage_code.strip_prefix('W')?)
        };

        (!rail_code.is_empty()).then_some((flow, rail_code))
    }

    fn matches_current_code(
        self,
        current_code: &str,
        rail_code: &str,
    ) -> bool {
        let prefix = match self {
            Self::Charging => 'I',
            Self::Discharging => 'Q',
        };
        current_code.strip_prefix(prefix) == Some(rail_code)
    }
}

pub fn rail_energy(
    voltage: &[Sensor],
    current: &[Sensor],
    elapsed: Duration,
) -> Option<Joules> {
    const MAX_PLAUSIBLE_VOLT_AMPS: f64 = 1000.0;
    let is_battery_rail = |sensor: &&Sensor| matches!(sensor.component, Component::Charger | Component::Battery);

    let mut charging_volt_amps = 0f64;
    let mut discharging_volt_amps = 0f64;
    for voltage_sensor in voltage.iter().filter(is_battery_rail) {
        let Some((voltage_area, voltage_code)) = voltage_sensor.name.rsplit_once(' ') else {
            continue;
        };
        let Some((flow, rail_code)) = PowerFlow::parse_voltage_code(voltage_code) else {
            continue;
        };
        for current_sensor in current.iter().filter(is_battery_rail) {
            let Some((current_area, current_code)) = current_sensor.name.rsplit_once(' ') else {
                continue;
            };
            if current_area == voltage_area && flow.matches_current_code(current_code, rail_code) {
                let volt_amps = (voltage_sensor.value * current_sensor.value).abs();
                if (0.0..=MAX_PLAUSIBLE_VOLT_AMPS).contains(&volt_amps) {
                    match flow {
                        PowerFlow::Charging => charging_volt_amps += volt_amps,
                        PowerFlow::Discharging => discharging_volt_amps += volt_amps,
                    }
                }
            }
        }
    }

    // Charging sensors stay nonzero on USB, so discharge sensors are only a fallback.
    let total_volt_amps = if charging_volt_amps > 0.0 {
        charging_volt_amps
    } else {
        discharging_volt_amps
    };
    (total_volt_amps > 0.0).then_some(Joules((total_volt_amps * elapsed.as_secs_f64()) as f32))
}
