use obfstr::obfstr;

pub(crate) struct ResidencyState {
    pub(crate) name: String,
    pub(crate) residency: i64,
}

pub(crate) struct ChannelSample {
    pub(crate) group: String,
    pub(crate) subgroup: String,
    pub(crate) name: String,
    pub(crate) unit: String,
    pub(crate) integer_value: i64,
    pub(crate) states: Vec<ResidencyState>,
}

pub(crate) fn cpu_clusters(
    channels: &[ChannelSample],
    ecpu_frequencies: &[u32],
    pcpu_frequencies: &[u32],
) -> ((u32, f32), (u32, f32)) {
    let mut ecpu_readings = Vec::new();
    let mut pcpu_readings = Vec::new();
    for channel in channels {
        if channel.group == obfstr!("CPU Stats") && channel.subgroup == obfstr!("CPU Core Performance States") {
            if channel.name.starts_with(obfstr!("PCPU")) {
                pcpu_readings.push(calculate_frequency(&channel.states, pcpu_frequencies));
            } else if channel.name.starts_with(obfstr!("ECPU")) || channel.name.starts_with(obfstr!("MCPU")) {
                ecpu_readings.push(calculate_frequency(&channel.states, ecpu_frequencies));
            }
        }
    }
    ecpu_readings.retain(|&(_, percent)| percent > 0.0);
    (
        average_cluster_frequency(&ecpu_readings, ecpu_frequencies),
        average_cluster_frequency(&pcpu_readings, pcpu_frequencies),
    )
}

pub(crate) fn gpu_frequency(
    channels: &[ChannelSample],
    gpu_frequencies: &[u32],
) -> (u32, f32) {
    for channel in channels {
        if channel.group == obfstr!("GPU Stats")
            && channel.subgroup == obfstr!("GPU Performance States")
            && channel.name == obfstr!("GPUPH")
            && gpu_frequencies.len() > 1
        {
            return calculate_frequency(&channel.states, &gpu_frequencies[1..]);
        }
    }
    (0, 0.0)
}

pub(crate) fn ane_active_percent(channels: &[ChannelSample]) -> f32 {
    let mut active = 0f32;
    for channel in channels {
        if channel.group == obfstr!("PMP")
            && channel.subgroup.contains(obfstr!("Floor"))
            && (channel.name == obfstr!("ANE-AF-BW") || channel.name == obfstr!("ANE-DCS-BW"))
        {
            active = active.max(residency_active_percent(&channel.states));
        }
    }
    active
}

pub(crate) fn dram_bandwidth(
    channels: &[ChannelSample],
    window_milliseconds: u64,
) -> (f32, f32) {
    let mut read_bytes = 0f64;
    let mut write_bytes = 0f64;
    let mut read_histogram = 0f32;
    let mut write_histogram = 0f32;
    for channel in channels {
        if channel.group == obfstr!("AMC Stats") {
            let bytes = channel.integer_value as f64;
            if bytes <= 0.0 {
                continue;
            }
            let aggregate = strip_die_prefix(&channel.name);
            if aggregate == obfstr!("DCS RD") {
                read_bytes += bytes;
            } else if aggregate == obfstr!("DCS WR") {
                write_bytes += bytes;
            }
        } else if channel.group == obfstr!("PMP") && channel.subgroup == obfstr!("DRAM BW") {
            let gbps = residency_weighted_gbps(&channel.states);
            match dram_flow(&channel.name) {
                Some(true) => read_histogram = read_histogram.max(gbps),
                Some(false) => write_histogram = write_histogram.max(gbps),
                None => {},
            }
        }
    }
    let window_seconds = (window_milliseconds as f64 / 1000.0).max(0.001);
    let to_gbps = |bytes: f64| (bytes / window_seconds / 1e9) as f32;
    let read = if read_bytes > 0.0 {
        to_gbps(read_bytes)
    } else {
        read_histogram
    };
    let write = if write_bytes > 0.0 {
        to_gbps(write_bytes)
    } else {
        write_histogram
    };
    (read, write)
}

pub(crate) fn energy_totals(channels: &[ChannelSample]) -> EnergyTotals {
    let mut totals = EnergyTotals::default();
    for channel in channels {
        if channel.group == obfstr!("Energy Model") {
            totals.accumulate(&channel.name, channel.integer_value, &channel.unit);
        }
    }
    totals
}

#[derive(Default, Clone, Copy)]
pub(crate) struct EnergyTotals {
    pub(crate) cpu: f64,
    pub(crate) gpu: f64,
    pub(crate) ane: f64,
    pub(crate) ram: f64,
}

impl EnergyTotals {
    fn accumulate(
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

fn calculate_frequency(
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

fn average_cluster_frequency(
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

fn residency_active_percent(states: &[ResidencyState]) -> f32 {
    let total: f64 = states.iter().map(|state| state.residency as f64).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let active: f64 =
        states.iter().filter(|state| !is_idle_state(&state.name)).map(|state| state.residency as f64).sum();
    (active / total * 100.0) as f32
}

fn residency_weighted_gbps(states: &[ResidencyState]) -> f32 {
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

fn strip_die_prefix(channel: &str) -> &str {
    let Some(rest) = channel.strip_prefix(obfstr!("DIE")) else {
        return channel;
    };
    let rest = rest.trim_start_matches(|character: char| character.is_ascii_digit());
    rest.strip_prefix(' ').unwrap_or(channel)
}

fn dram_flow(channel: &str) -> Option<bool> {
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
