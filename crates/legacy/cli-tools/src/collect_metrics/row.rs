use serde::Serialize;

use super::{device_info::DeviceInfo, measurement::Measurement, resolved_target::ResolvedTarget};
use crate::types::SourceMode;

/// One CSV record: a single measured prefill/generate pass on one model.
#[derive(Serialize)]
pub struct Row {
    pub os: String,
    pub chip: String,
    pub ram_total_bytes: u64,
    pub gpu_cores: u8,
    pub source: SourceMode,
    pub model_id: String,
    pub prefill_tokens: usize,
    pub generate_tokens: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub total_ms: f64,
    pub decode_tokens_per_second: f64,
    pub energy_total_j: f64,
    pub energy_cpu_j: f64,
    pub energy_gpu_j: f64,
    pub energy_ram_j: f64,
    pub avg_watts_total: f64,
    pub avg_watts_cpu: f64,
    pub avg_watts_gpu: f64,
    pub avg_watts_ram: f64,
    pub avg_joules_per_prefill_token: f64,
    pub avg_joules_per_decode_token: f64,
    pub dram_read_bytes: u64,
    pub dram_write_bytes: u64,
    pub dram_read_gbps: f64,
    pub dram_write_gbps: f64,
}

impl Row {
    /// Returns `None` when the measurement window produced no power reading.
    pub fn measured(
        device: &DeviceInfo,
        target: &ResolvedTarget,
        prefill: usize,
        generate: usize,
        measurement: &Measurement,
    ) -> Option<Self> {
        let reading = measurement.reading.as_ref()?;
        let prefill_ms = measurement.prefill_ms;
        let decode_ms = measurement.decode_ms;
        let total_ms = prefill_ms + decode_ms;
        let energy_total_j = reading.energy_total_j();
        let per_second = |joules: f64| {
            if total_ms > 0.0 {
                joules * 1000.0 / total_ms
            } else {
                0.0
            }
        };
        let share = |phase_ms: f64| {
            if total_ms > 0.0 {
                energy_total_j * (phase_ms / total_ms)
            } else {
                0.0
            }
        };
        let per_token = |joules: f64, tokens: usize| {
            if tokens > 0 {
                joules / tokens as f64
            } else {
                0.0
            }
        };

        Some(Self {
            os: device.os.clone(),
            chip: device.chip.clone(),
            ram_total_bytes: device.ram_total_bytes,
            gpu_cores: device.gpu_cores,
            source: target.source,
            model_id: target.id.clone(),
            prefill_tokens: prefill,
            generate_tokens: generate,
            prefill_ms,
            decode_ms,
            total_ms,
            decode_tokens_per_second: if decode_ms > 0.0 {
                measurement.decode_tokens as f64 * 1000.0 / decode_ms
            } else {
                0.0
            },
            energy_total_j,
            energy_cpu_j: reading.cpu_j,
            energy_gpu_j: reading.gpu_j,
            energy_ram_j: reading.ram_j,
            avg_watts_total: per_second(energy_total_j),
            avg_watts_cpu: per_second(reading.cpu_j),
            avg_watts_gpu: per_second(reading.gpu_j),
            avg_watts_ram: per_second(reading.ram_j),
            avg_joules_per_prefill_token: per_token(share(prefill_ms), prefill),
            avg_joules_per_decode_token: per_token(share(decode_ms), measurement.decode_tokens),
            dram_read_bytes: reading.dram_read_bytes,
            dram_write_bytes: reading.dram_write_bytes,
            dram_read_gbps: reading.dram_read_gbps,
            dram_write_gbps: reading.dram_write_gbps,
        })
    }
}
