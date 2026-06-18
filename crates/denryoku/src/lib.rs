mod component;
mod sensor;
mod sys;

#[cfg(target_vendor = "apple")]
mod client;

#[cfg(target_vendor = "apple")]
pub use client::Sampler;
pub use component::{Component, classify};
pub use sensor::{Sensor, SensorKind, current_sensors, thermal_sensors, voltage_sensors};

#[cfg(target_vendor = "apple")]
pub fn sensors(kind: SensorKind) -> Vec<Sensor> {
    client::collect(kind)
}

#[cfg(target_vendor = "apple")]
pub fn is_available() -> bool {
    client::is_available()
}

#[cfg(not(target_vendor = "apple"))]
pub fn sensors(_kind: SensorKind) -> Vec<Sensor> {
    Vec::new()
}

#[cfg(not(target_vendor = "apple"))]
pub fn is_available() -> bool {
    false
}

/// Sensor sampler (see the Apple-only implementation); unavailable here.
#[cfg(not(target_vendor = "apple"))]
pub struct Sampler;

#[cfg(not(target_vendor = "apple"))]
impl Sampler {
    pub fn new(_kind: SensorKind) -> Option<Self> {
        None
    }

    pub fn sample(&mut self) -> &[Sensor] {
        &[]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queries_do_not_panic() {
        let _ = is_available();
        for kind in [SensorKind::Temperature, SensorKind::Voltage, SensorKind::Current] {
            for sensor in sensors(kind) {
                assert_eq!(sensor.kind, kind);
                assert!(sensor.value.is_finite());
            }
            if let Some(mut sampler) = Sampler::new(kind) {
                for sensor in sampler.sample() {
                    assert_eq!(sensor.kind, kind);
                    assert!(sensor.value.is_finite());
                }
            }
        }
    }
}
