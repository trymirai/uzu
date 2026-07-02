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
    pub prefill_energy_joules: Option<f64>,
    pub decode_energy_joules: Option<f64>,
}

#[cfg(target_vendor = "apple")]
impl PowerStats {
    pub(crate) fn from_energy_readings(
        total: keisoku::EnergyReading,
        prefill: Option<keisoku::EnergyReading>,
        decode: Option<keisoku::EnergyReading>,
    ) -> Option<Self> {
        let energy_joules = total.energy.package.value() as f64;
        if energy_joules <= 0.0 {
            return None;
        }
        let average_package_watts = total.average_power.package.value() as f64;
        Some(Self {
            samples_count: 1,
            average_cpu_watts: total.average_power.cpu.value() as f64,
            average_gpu_watts: total.average_power.gpu.value() as f64,
            average_gpu_sram_watts: total.average_power.gpu_sram.value() as f64,
            average_ane_watts: total.average_power.ane.value() as f64,
            average_ram_watts: total.average_power.ram.value() as f64,
            average_total_watts: total.average_power.total.value() as f64,
            average_package_watts,
            max_package_watts: average_package_watts,
            energy_joules,
            prefill_energy_joules: prefill.map(|reading| reading.energy.package.value() as f64),
            decode_energy_joules: decode.map(|reading| reading.energy.package.value() as f64),
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
