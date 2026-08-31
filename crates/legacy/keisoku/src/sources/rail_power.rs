use crate::{component::Component, sensor::Sensor, units::Watts};

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

pub(crate) fn rail_power(
    voltage: &[Sensor],
    current: &[Sensor],
) -> Option<Watts> {
    const MAX_PLAUSIBLE_WATTS: f64 = 1000.0;
    let is_battery_rail = |sensor: &&Sensor| matches!(sensor.component, Component::Charger | Component::Battery);

    let mut charging_watts = 0f64;
    let mut discharging_watts = 0f64;
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
                let watts = (voltage_sensor.value * current_sensor.value).abs();
                if (0.0..=MAX_PLAUSIBLE_WATTS).contains(&watts) {
                    match flow {
                        PowerFlow::Charging => charging_watts += watts,
                        PowerFlow::Discharging => discharging_watts += watts,
                    }
                }
            }
        }
    }

    // CONTEXT: Charger input and battery discharge sensors can both be nonzero
    // while plugged in. Preserve V/I charging power and use W/Q only as the
    // battery-only fallback observed when the device is unplugged.
    let total_watts = if charging_watts > 0.0 {
        charging_watts
    } else {
        discharging_watts
    };
    (total_watts > 0.0).then_some(Watts(total_watts as f32))
}
