use crate::{component::Component, sensor::Sensor, units::Watts};

pub(crate) fn rail_power(
    voltage: &[Sensor],
    current: &[Sensor],
) -> Option<Watts> {
    const MAX_PLAUSIBLE_WATTS: f64 = 1000.0;
    let split_area_code = |name: &str| name.rsplit_once(' ').map(|(area, code)| (area.to_owned(), code.to_owned()));
    let is_battery_rail = |sensor: &&Sensor| matches!(sensor.component, Component::Charger | Component::Battery);

    let mut total_watts = 0f64;
    for voltage_sensor in voltage.iter().filter(is_battery_rail) {
        let Some((voltage_area, voltage_code)) = split_area_code(&voltage_sensor.name) else {
            continue;
        };
        let Some(rail_code) = voltage_code.strip_prefix('V').filter(|code| !code.is_empty()) else {
            continue;
        };
        for current_sensor in current.iter().filter(is_battery_rail) {
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
