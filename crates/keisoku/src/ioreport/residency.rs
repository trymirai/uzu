use obfstr::obfstr;

pub(super) struct ResidencyState {
    pub name: String,
    pub residency: i64,
}

pub(super) fn residency_active_percent(states: &[ResidencyState]) -> f32 {
    let total: f64 = states.iter().map(|state| state.residency as f64).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let active: f64 =
        states.iter().filter(|state| !is_idle_state(&state.name)).map(|state| state.residency as f64).sum();
    (active / total * 100.0) as f32
}

pub(super) fn residency_weighted_gbps(states: &[ResidencyState]) -> f32 {
    let mut weighted = 0f64;
    let mut total = 0f64;
    for state in states {
        weighted += parse_leading_number(&state.name) * (state.residency as f64);
        total += state.residency as f64;
    }
    if total <= 0.0 {
        0.0
    } else {
        (weighted / total) as f32
    }
}

pub(super) fn is_idle_state(name: &str) -> bool {
    name == obfstr!("OFF")
        || name == obfstr!("IDLE")
        || name == obfstr!("DOWN")
        || name == obfstr!("SLEEP")
        || name == obfstr!("VMIN")
        || name == obfstr!("F1")
        || name == obfstr!("0%")
}

fn parse_leading_number(name: &str) -> f64 {
    let digits: String =
        name.trim().chars().take_while(|character| character.is_ascii_digit() || *character == '.').collect();
    digits.parse().unwrap_or(0.0)
}
