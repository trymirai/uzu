use super::reading::Reading;
use crate::{component::Component, metrics::Temperatures, sensor::Sensor, sources::Sources, units::Celsius};

pub struct Temps;

impl Reading for Temps {
    type Value = Option<Temperatures>;

    fn read(sources: &mut Sources) -> Option<Temperatures> {
        let sensors = sources.temperature_sensors();
        (!sensors.is_empty()).then(|| temperatures_from(&sensors))
    }
}

fn temperatures_from(sensors: &[Sensor]) -> Temperatures {
    let average_of = |components: &[Component]| {
        let values: Vec<f32> = sensors
            .iter()
            .filter(|sensor| components.contains(&sensor.component) && (1.0..150.0).contains(&sensor.value))
            .map(|sensor| sensor.value as f32)
            .collect();
        (!values.is_empty()).then(|| Celsius(average(&values)))
    };
    Temperatures {
        cpu_average: average_of(&[Component::Cpu, Component::Soc]),
        gpu_average: average_of(&[Component::Gpu]),
    }
}

fn average(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}
