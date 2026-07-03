use super::residency::{ResidencyState, is_idle_state};

pub(super) fn calculate_frequency(
    states: &[ResidencyState],
    frequencies: &[u32],
) -> (u32, f32) {
    if states.len() <= frequencies.len() || frequencies.is_empty() {
        return (0, 0.0);
    }

    let Some(offset) = states.iter().position(|state| !is_idle_state(&state.name)) else {
        return (0, 0.0);
    };
    let active: f64 = states.iter().skip(offset).map(|state| state.residency as f64).sum();
    let total: f64 = states.iter().map(|state| state.residency as f64).sum();

    let mut average_frequency = 0f64;
    for (index, &frequency) in frequencies.iter().enumerate() {
        let Some(state) = states.get(index + offset) else {
            break;
        };
        let percent = divide_or_zero(state.residency as f64, active);
        average_frequency += percent * frequency as f64;
    }
    let usage_ratio = divide_or_zero(active, total);
    let minimum_frequency = *frequencies.first().unwrap() as f64;
    let maximum_frequency = *frequencies.last().unwrap() as f64;
    let fraction_of_max = (average_frequency.max(minimum_frequency) * usage_ratio) / maximum_frequency;
    (average_frequency as u32, fraction_of_max as f32)
}

pub(super) fn average_cluster_frequency(
    readings: &[(u32, f32)],
    frequencies: &[u32],
) -> (u32, f32) {
    if readings.is_empty() || frequencies.is_empty() {
        return (0, 0.0);
    }
    let average_frequency =
        divide_or_zero(readings.iter().map(|reading| reading.0 as f32).sum::<f32>(), readings.len() as f32);
    let average_percent = divide_or_zero(readings.iter().map(|reading| reading.1).sum::<f32>(), readings.len() as f32);
    let minimum_frequency = *frequencies.first().unwrap() as f32;
    (average_frequency.max(minimum_frequency) as u32, average_percent)
}

pub(super) fn divide_or_zero<T: core::ops::Div<Output = T> + Default + PartialEq>(
    numerator: T,
    denominator: T,
) -> T {
    let zero = T::default();
    if denominator == zero {
        zero
    } else {
        numerator / denominator
    }
}
