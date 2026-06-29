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
        // Each accumulator integrates watts over time (watts × seconds); the time-weighted
        // average is the integral divided by total_elapsed_secs.
        let mut cpu_joules = 0.0_f64;
        let mut gpu_joules = 0.0_f64;
        let mut gpu_sram_joules = 0.0_f64;
        let mut ane_joules = 0.0_f64;
        let mut ram_joules = 0.0_f64;
        let mut total_joules = 0.0_f64;
        let mut energy_joules = 0.0_f64;
        let mut max_package_watts = 0.0_f64;
        let mut previous_elapsed_secs = 0.0_f64;
        let mut total_elapsed_secs = 0.0_f64;

        for snapshot in &session.snapshots {
            // elapsed is stamped after each sample, so consecutive gaps partition the
            // recording timeline without overlap.
            let elapsed_secs = snapshot.elapsed.value() as f64 / 1000.0;
            let window_secs = (elapsed_secs - previous_elapsed_secs).max(0.0);
            previous_elapsed_secs = elapsed_secs;

            // Prefer SoC IOReport power (macOS); fall back to HID rail power on iOS.
            let wall_watts = match snapshot.power.as_ref() {
                Some(power) => {
                    cpu_joules += power.cpu.value() as f64 * window_secs;
                    gpu_joules += power.gpu.value() as f64 * window_secs;
                    gpu_sram_joules += power.gpu_sram.value() as f64 * window_secs;
                    ane_joules += power.ane.value() as f64 * window_secs;
                    ram_joules += power.ram.value() as f64 * window_secs;
                    total_joules += power.total.value() as f64 * window_secs;
                    power.package.value() as f64
                },
                None => match snapshot.rail_power() {
                    Some(rail_watts) => {
                        let rail_watts = rail_watts.value() as f64;
                        total_joules += rail_watts * window_secs;
                        rail_watts
                    },
                    None => continue,
                },
            };
            samples_count += 1;
            max_package_watts = max_package_watts.max(wall_watts);
            energy_joules += wall_watts * window_secs;
            total_elapsed_secs = elapsed_secs;
        }

        if samples_count == 0 || total_elapsed_secs <= 0.0 {
            return None;
        }

        Some(Self {
            samples_count,
            average_cpu_watts: cpu_joules / total_elapsed_secs,
            average_gpu_watts: gpu_joules / total_elapsed_secs,
            average_gpu_sram_watts: gpu_sram_joules / total_elapsed_secs,
            average_ane_watts: ane_joules / total_elapsed_secs,
            average_ram_watts: ram_joules / total_elapsed_secs,
            average_total_watts: total_joules / total_elapsed_secs,
            average_package_watts: energy_joules / total_elapsed_secs,
            max_package_watts,
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
