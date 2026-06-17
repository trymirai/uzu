mod component;
mod sensor;
mod sys;

#[cfg(target_vendor = "apple")]
mod client;

pub use component::{Component, classify};
pub use sensor::{Sensor, SensorKind, current_sensors, thermal_sensors, voltage_sensors};

#[cfg(target_vendor = "apple")]
pub fn sensors(kind: SensorKind) -> Vec<Sensor> {
    client::collect(kind)
}

#[cfg(not(target_vendor = "apple"))]
pub fn sensors(_kind: SensorKind) -> Vec<Sensor> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queries_do_not_panic() {
        for kind in [SensorKind::Temperature, SensorKind::Voltage, SensorKind::Current] {
            for sensor in sensors(kind) {
                assert_eq!(sensor.kind, kind);
                assert!(sensor.value.is_finite());
            }
        }
    }
}
