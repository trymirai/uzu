use std::path::Path;

use anyhow::Result;
use serde::Serialize;

use super::{SourceMode, workload::Measurement};

pub struct DeviceInfo {
    pub os: String,
    pub chip: String,
    pub ram_total_bytes: u64,
    pub gpu_cores: u8,
}

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
}

impl Row {
    pub fn measured(
        device: &DeviceInfo,
        source: SourceMode,
        model_id: &str,
        prefill: usize,
        generate: usize,
        measurement: &Measurement,
    ) -> Option<Self> {
        let reading = measurement.reading.as_ref()?;
        let prefill_ms = measurement.prefill_ms;
        let decode_ms = measurement.decode_ms;
        let total_ms = prefill_ms + decode_ms;
        let total_seconds = total_ms / 1000.0;
        let energy_total_j = reading.energy_total_j;
        let prefill_energy_j = if total_ms > 0.0 {
            energy_total_j * (prefill_ms / total_ms)
        } else {
            0.0
        };
        let decode_energy_j = if total_ms > 0.0 {
            energy_total_j * (decode_ms / total_ms)
        } else {
            0.0
        };

        Some(Self {
            os: device.os.clone(),
            chip: device.chip.clone(),
            ram_total_bytes: device.ram_total_bytes,
            gpu_cores: device.gpu_cores,
            source,
            model_id: model_id.to_string(),
            prefill_tokens: prefill,
            generate_tokens: generate,
            prefill_ms,
            decode_ms,
            total_ms,
            decode_tokens_per_second: if decode_ms > 0.0 {
                measurement.decode_tokens as f64 / (decode_ms / 1000.0)
            } else {
                0.0
            },
            energy_total_j,
            energy_cpu_j: reading.cpu_watts * total_seconds,
            energy_gpu_j: reading.gpu_watts * total_seconds,
            energy_ram_j: reading.ram_watts * total_seconds,
            avg_watts_total: reading.total_watts,
            avg_watts_cpu: reading.cpu_watts,
            avg_watts_gpu: reading.gpu_watts,
            avg_watts_ram: reading.ram_watts,
            avg_joules_per_prefill_token: if prefill > 0 {
                prefill_energy_j / prefill as f64
            } else {
                0.0
            },
            avg_joules_per_decode_token: if measurement.decode_tokens > 0 {
                decode_energy_j / measurement.decode_tokens as f64
            } else {
                0.0
            },
        })
    }
}

pub struct Report {
    writer: csv::Writer<std::fs::File>,
}

impl Report {
    pub fn new(path: &Path) -> Result<Self> {
        let mut writer = csv::WriterBuilder::new().has_headers(false).from_path(path)?;
        writer.write_record([
            "os",
            "chip",
            "ram_total_bytes",
            "gpu_cores",
            "source",
            "model_id",
            "prefill_tokens",
            "generate_tokens",
            "prefill_ms",
            "decode_ms",
            "total_ms",
            "decode_tokens_per_second",
            "energy_total_j",
            "energy_cpu_j",
            "energy_gpu_j",
            "energy_ram_j",
            "avg_watts_total",
            "avg_watts_cpu",
            "avg_watts_gpu",
            "avg_watts_ram",
            "avg_joules_per_prefill_token",
            "avg_joules_per_decode_token",
        ])?;
        writer.flush()?;
        Ok(Self {
            writer,
        })
    }

    pub fn write(
        &mut self,
        row: &Row,
    ) -> Result<()> {
        self.writer.serialize(row)?;
        self.writer.flush()?;
        Ok(())
    }
}
