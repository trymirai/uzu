use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PowerStats {
    pub samples_count: u64,
    pub average_cpu_watts: f64,
    pub average_gpu_watts: f64,
    pub average_gpu_sram_watts: f64,
    pub average_ane_watts: f64,
    pub average_ram_watts: f64,
    pub average_total_watts: f64,
    pub average_package_watts: f64,
    pub max_package_watts: f64,
    pub energy_joules: f64,
}

#[cfg(target_vendor = "apple")]
impl PowerStats {
    pub(crate) fn from_keisoku_session(session: &keisoku::Session) -> Option<Self> {
        let mut samples_count = 0_u64;
        // Joule accumulators: watts × window_secs for each component.
        // Dividing by total_elapsed_secs gives the time-weighted average, so
        // average_package_watts × total_elapsed_secs = energy_joules exactly.
        let mut cpu_j = 0.0_f64;
        let mut gpu_j = 0.0_f64;
        let mut gpu_sram_j = 0.0_f64;
        let mut ane_j = 0.0_f64;
        let mut ram_j = 0.0_f64;
        let mut total_j = 0.0_f64;
        let mut energy_joules = 0.0_f64;
        let mut max_package = 0.0_f64;
        let mut previous_elapsed_secs = 0.0_f64;
        let mut total_elapsed_secs = 0.0_f64;

        for snapshot in &session.snapshots {
            let elapsed_secs = snapshot.elapsed.value() as f64 / 1000.0;
            // window_secs = duration this snapshot's readings were observed over.
            // elapsed is stamped after each collector.sample() call, so consecutive
            // gaps partition the recording timeline without overlap.
            let window_secs = (elapsed_secs - previous_elapsed_secs).max(0.0);
            previous_elapsed_secs = elapsed_secs;

            // Prefer SoC IOReport power (macOS); fall back to HID rail power on iOS.
            let wall_watts = match snapshot.power.as_ref() {
                Some(power) => {
                    cpu_j += power.cpu.value() as f64 * window_secs;
                    gpu_j += power.gpu.value() as f64 * window_secs;
                    gpu_sram_j += power.gpu_sram.value() as f64 * window_secs;
                    ane_j += power.ane.value() as f64 * window_secs;
                    ram_j += power.ram.value() as f64 * window_secs;
                    total_j += power.total.value() as f64 * window_secs;
                    power.package.value() as f64
                },
                None => match snapshot.rail_power() {
                    Some(watts) => {
                        let watts = watts.value() as f64;
                        total_j += watts * window_secs;
                        watts
                    },
                    None => continue,
                },
            };
            samples_count += 1;
            max_package = max_package.max(wall_watts);
            energy_joules += wall_watts * window_secs;
            total_elapsed_secs = elapsed_secs;
        }

        if samples_count == 0 {
            return None;
        }

        let t = total_elapsed_secs;
        Some(Self {
            samples_count,
            average_cpu_watts: cpu_j / t,
            average_gpu_watts: gpu_j / t,
            average_gpu_sram_watts: gpu_sram_j / t,
            average_ane_watts: ane_j / t,
            average_ram_watts: ram_j / t,
            average_total_watts: total_j / t,
            average_package_watts: energy_joules / t,
            max_package_watts: max_package,
            energy_joules,
        })
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RunStats {
    pub count: u64,
    pub average_duration: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StepStats {
    pub duration: f64,
    pub suffix_length: u64,
    pub tokens_count: u64,
    pub tokens_per_second: f64,
    pub processed_tokens_per_second: f64,
    pub model_run: RunStats,
    pub run: Option<RunStats>,
    pub speculator_proposed: u64,
    pub speculator_accepted: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TotalStats {
    pub duration: f64,
    pub tokens_count_input: u64,
    pub tokens_count_output: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Stats {
    pub prefill_stats: StepStats,
    pub generate_stats: Option<StepStats>,
    pub total_stats: TotalStats,
    pub memory_used_bytes: Option<u64>,
    pub power_stats: Option<PowerStats>,
}
