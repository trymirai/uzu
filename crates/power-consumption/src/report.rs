use std::path::Path;

use anyhow::Result;
use serde::Serialize;
use shoji::types::model::Model;

use crate::workload::Measurement;

#[derive(Clone)]
pub struct DeviceInfo {
    pub os: String,
    pub chip: String,
    pub ram_total_bytes: u64,
    pub gpu_cores: u8,
}

#[derive(Clone)]
pub struct ModelMeta {
    pub id: String,
}

impl ModelMeta {
    pub fn from_model(model: &Model) -> Self {
        Self {
            id: model.identifier.clone(),
        }
    }
}

#[derive(Serialize)]
pub struct Row {
    pub os: String,
    pub chip: String,
    pub ram_total_bytes: u64,
    pub gpu_cores: u8,
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
    pub energy_ane_j: f64,
    pub energy_ram_j: f64,
    pub avg_watts_total: f64,
    pub avg_watts_cpu: f64,
    pub avg_watts_gpu: f64,
    pub avg_watts_ane: f64,
    pub avg_watts_ram: f64,
    pub avg_joules_per_prefill_token: f64,
    pub avg_joules_per_decode_token: f64,
}

impl Row {
    pub fn measured(
        device: &DeviceInfo,
        meta: &ModelMeta,
        prefill: usize,
        generate: usize,
        measurement: &Measurement,
    ) -> Option<Self> {
        let reading = measurement.reading.as_ref()?;
        let prefill_ms = measurement.prefill_ms;
        let decode_ms = measurement.decode_ms;
        let total_ms = prefill_ms + decode_ms;
        let energy_total_j = reading.energy.total().value() as f64;
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
            model_id: meta.id.clone(),
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
            energy_cpu_j: reading.energy.cpu.value() as f64,
            energy_gpu_j: reading.energy.gpu.value() as f64,
            energy_ane_j: reading.energy.ane.value() as f64,
            energy_ram_j: reading.energy.ram.value() as f64,
            avg_watts_total: reading.average_power.total().value() as f64,
            avg_watts_cpu: reading.average_power.cpu.value() as f64,
            avg_watts_gpu: reading.average_power.gpu.value() as f64,
            avg_watts_ane: reading.average_power.ane.value() as f64,
            avg_watts_ram: reading.average_power.ram.value() as f64,
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
        Ok(Self {
            writer: csv::Writer::from_path(path)?,
        })
    }

    pub fn write(
        &mut self,
        row: &Row,
    ) -> Result<()> {
        self.writer.serialize(row)?;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}
