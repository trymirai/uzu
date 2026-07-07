use obfstr::obfstr;

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum GroupId {
    EnergyModel,
    CpuStats,
    GpuStats,
    AmcStats,
    Pmp,
    #[default]
    Other,
}

impl GroupId {
    pub(crate) fn classify(group: &str) -> GroupId {
        if group == obfstr!("Energy Model") {
            GroupId::EnergyModel
        } else if group == obfstr!("CPU Stats") {
            GroupId::CpuStats
        } else if group == obfstr!("GPU Stats") {
            GroupId::GpuStats
        } else if group == obfstr!("AMC Stats") {
            GroupId::AmcStats
        } else if group == obfstr!("PMP") {
            GroupId::Pmp
        } else {
            GroupId::Other
        }
    }
}

#[derive(Default)]
pub(crate) struct ResidencyState {
    pub(crate) name: String,
    pub(crate) residency: i64,
}

/// One decoded IOReport channel. Only the fields a channel's group actually
/// uses are populated (see `ioreport::channel`), so byte-counter channels carry
/// no residency states and energy channels carry no subgroup.
#[derive(Default)]
pub struct RawChannel {
    pub(crate) group: GroupId,
    pub(crate) subgroup: String,
    pub(crate) name: String,
    pub(crate) unit: String,
    pub(crate) integer_value: i64,
    pub(crate) states: Vec<ResidencyState>,
}

/// Borrowed SoC frequency tables — the only context CPU/GPU usage metrics need.
#[derive(Default, Clone, Copy)]
pub struct FrequencyTables<'a> {
    pub(crate) ecpu: &'a [u32],
    pub(crate) pcpu: &'a [u32],
    pub(crate) gpu: &'a [u32],
    pub(crate) ecpu_cores: u8,
    pub(crate) pcpu_cores: u8,
}

#[derive(Default, Clone, Copy)]
pub struct EnergyTotals {
    pub(crate) cpu: f64,
    pub(crate) gpu: f64,
    pub(crate) ane: f64,
    pub(crate) ram: f64,
}

impl EnergyTotals {
    pub(crate) fn accumulate(
        &mut self,
        name: &str,
        value: i64,
        unit: &str,
    ) {
        let joules = joules(value, unit);
        if name == obfstr!("GPU Energy") {
            self.gpu += joules;
        } else if name.ends_with(obfstr!("CPU Energy")) {
            self.cpu += joules;
        } else if name.starts_with(obfstr!("ANE")) {
            self.ane += joules;
        } else if name.starts_with(obfstr!("DRAM"))
            || name.starts_with(obfstr!("DCS"))
            || name.starts_with(obfstr!("AMCC"))
        {
            self.ram += joules;
        }
    }

    pub(crate) fn total(&self) -> f64 {
        self.cpu + self.gpu + self.ane + self.ram
    }
}

pub(crate) fn calculate_frequency(
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

pub(crate) fn average_cluster_frequency(
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

pub(crate) fn divide_or_zero<T: core::ops::Div<Output = T> + Default + PartialEq>(
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

pub(crate) fn residency_active_percent(states: &[ResidencyState]) -> f32 {
    let total: f64 = states.iter().map(|state| state.residency as f64).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let active: f64 =
        states.iter().filter(|state| !is_idle_state(&state.name)).map(|state| state.residency as f64).sum();
    (active / total * 100.0) as f32
}

pub(crate) fn residency_weighted_gbps(states: &[ResidencyState]) -> f32 {
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

pub(crate) fn strip_die_prefix(channel: &str) -> &str {
    let Some(rest) = channel.strip_prefix(obfstr!("DIE")) else {
        return channel;
    };
    let rest = rest.trim_start_matches(|character: char| character.is_ascii_digit());
    rest.strip_prefix(' ').unwrap_or(channel)
}

pub(crate) fn dram_flow(channel: &str) -> Option<bool> {
    if channel.contains(obfstr!("RD+WR")) || channel.ends_with(obfstr!("RW")) {
        None
    } else if channel.contains(obfstr!("WR")) {
        Some(false)
    } else if channel.contains(obfstr!("RD")) {
        Some(true)
    } else {
        None
    }
}

fn is_idle_state(name: &str) -> bool {
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

fn joules(
    energy: i64,
    unit: &str,
) -> f64 {
    let energy = energy as f64;
    let Some(prefix) = unit.trim().strip_suffix('J') else {
        return 0.0;
    };
    let scale = match prefix {
        "k" => 1e3,
        "" => 1.0,
        "m" => 1e-3,
        "u" | "µ" => 1e-6,
        "n" => 1e-9,
        "p" => 1e-12,
        _ => return 0.0,
    };
    energy * scale
}
