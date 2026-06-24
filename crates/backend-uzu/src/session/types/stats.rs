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
    pub(crate) fn from_keisoku_session(
        session: &keisoku::Session,
        duration: f64,
    ) -> Option<Self> {
        let mut samples_count = 0_u64;
        let mut cpu = 0.0;
        let mut gpu = 0.0;
        let mut gpu_sram = 0.0;
        let mut ane = 0.0;
        let mut ram = 0.0;
        let mut total = 0.0;
        let mut package = 0.0;
        let mut max_package = 0.0_f64;

        for snapshot in &session.snapshots {
            // Prefer SoC IOReport power (macOS); fall back to HID charger "wall" power on iOS (per-component stays 0).
            let wall_watts = match snapshot.power.as_ref() {
                Some(power) => {
                    cpu += power.cpu.value() as f64;
                    gpu += power.gpu.value() as f64;
                    gpu_sram += power.gpu_sram.value() as f64;
                    ane += power.ane.value() as f64;
                    ram += power.ram.value() as f64;
                    total += power.total.value() as f64;
                    power.package.value() as f64
                },
                None => match snapshot.rail_power() {
                    Some(watts) => {
                        let watts = watts.value() as f64;
                        total += watts;
                        watts
                    },
                    None => continue,
                },
            };
            samples_count += 1;
            package += wall_watts;
            max_package = max_package.max(wall_watts);
        }

        if samples_count == 0 {
            return None;
        }

        let sample_count = samples_count as f64;
        let average_package_watts = package / sample_count;
        Some(Self {
            samples_count,
            average_cpu_watts: cpu / sample_count,
            average_gpu_watts: gpu / sample_count,
            average_gpu_sram_watts: gpu_sram / sample_count,
            average_ane_watts: ane / sample_count,
            average_ram_watts: ram / sample_count,
            average_total_watts: total / sample_count,
            average_package_watts,
            max_package_watts: max_package,
            energy_joules: average_package_watts * duration.max(0.0),
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
