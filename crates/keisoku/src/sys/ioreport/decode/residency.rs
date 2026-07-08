use obfstr::obfstr;

use super::residency_state::ResidencyState;

pub(crate) fn residency_active_percent(states: &[ResidencyState]) -> f32 {
    let total: f64 = states.iter().map(|state| state.residency as f64).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let active: f64 =
        states.iter().filter(|state| !is_idle_state(&state.name)).map(|state| state.residency as f64).sum();
    (active / total * 100.0) as f32
}

pub(crate) fn is_idle_state(name: &str) -> bool {
    name == obfstr!("OFF")
        || name == obfstr!("IDLE")
        || name == obfstr!("DOWN")
        || name == obfstr!("SLEEP")
        || name == obfstr!("VMIN")
        || name == obfstr!("F1")
        || name == obfstr!("0%")
}
