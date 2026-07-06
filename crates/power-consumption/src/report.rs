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

#[derive(Clone, Default)]
pub struct ModelMeta {
    pub id: String,
    pub name: String,
    pub size_bytes: Option<i64>,
    pub quantization: String,
    pub num_layers: usize,
    pub model_dim: usize,
    pub hidden_dim: usize,
    pub vocab_size: usize,
}

impl ModelMeta {
    pub fn from_model(model: &Model) -> Self {
        Self {
            id: model.identifier.clone(),
            name: model.name(),
            size_bytes: model.properties.as_ref().map(|properties| properties.size),
            quantization: model.quantization.as_ref().map(|quantization| quantization.name()).unwrap_or_default(),
            ..Default::default()
        }
    }

    pub fn with_config(
        mut self,
        config_path: &Path,
    ) -> Self {
        let Ok(text) = std::fs::read_to_string(config_path) else {
            return self;
        };
        let Ok(config) = serde_json::from_str::<serde_json::Value>(&text) else {
            return self;
        };
        let decoder = &config["decoder_config"];
        let transformer = &decoder["transformer_config"];
        if let Some(vocab_size) = decoder["vocab_size"].as_u64() {
            self.vocab_size = vocab_size as usize;
        }
        if let Some(model_dim) = transformer["model_dim"].as_u64() {
            self.model_dim = model_dim as usize;
        }
        if let Some(hidden_dim) = transformer["hidden_dim"].as_u64() {
            self.hidden_dim = hidden_dim as usize;
        }
        if let Some(layers) = transformer["layer_configs"].as_array() {
            self.num_layers = layers.len();
        }
        self
    }
}

#[derive(Serialize, Default)]
pub struct Row {
    pub os: String,
    pub chip: String,
    pub ram_total_bytes: u64,
    pub gpu_cores: u8,
    pub backend: String,
    pub model_id: String,
    pub model_name: String,
    pub model_size_bytes: i64,
    pub quantization: String,
    pub num_layers: usize,
    pub model_dim: usize,
    pub hidden_dim: usize,
    pub vocab_size: usize,
    pub prefill_tokens: usize,
    pub generate_tokens: usize,
    pub repetition: usize,
    pub status: String,
    pub error: String,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub total_ms: f64,
    pub decode_tokens_per_second: f64,
    pub energy_total_j: f64,
    pub energy_cpu_j: f64,
    pub energy_gpu_j: f64,
    pub energy_gpu_sram_j: f64,
    pub energy_ane_j: f64,
    pub energy_ram_j: f64,
    pub avg_watts_total: f64,
    pub avg_watts_cpu: f64,
    pub avg_watts_gpu: f64,
    pub avg_watts_gpu_sram: f64,
    pub avg_watts_ane: f64,
    pub avg_watts_ram: f64,
}

impl Row {
    pub fn status(
        device: &DeviceInfo,
        meta: &ModelMeta,
        status: &str,
        error: &str,
    ) -> Self {
        Self {
            os: device.os.clone(),
            chip: device.chip.clone(),
            ram_total_bytes: device.ram_total_bytes,
            gpu_cores: device.gpu_cores,
            backend: "metal".to_string(),
            model_id: meta.id.clone(),
            model_name: meta.name.clone(),
            model_size_bytes: meta.size_bytes.unwrap_or(0),
            quantization: meta.quantization.clone(),
            num_layers: meta.num_layers,
            model_dim: meta.model_dim,
            hidden_dim: meta.hidden_dim,
            vocab_size: meta.vocab_size,
            status: status.to_string(),
            error: error.to_string(),
            ..Default::default()
        }
    }

    pub fn measured(
        device: &DeviceInfo,
        meta: &ModelMeta,
        prefill: usize,
        generate: usize,
        repetition: usize,
        measurement: &Measurement,
    ) -> Self {
        let mut row = Self::status(device, meta, "ok", "");
        row.prefill_tokens = prefill;
        row.generate_tokens = generate;
        row.repetition = repetition;
        row.prefill_ms = measurement.prefill_ms;
        row.decode_ms = measurement.decode_ms;
        row.total_ms = measurement.prefill_ms + measurement.decode_ms;
        row.decode_tokens_per_second = if measurement.decode_ms > 0.0 {
            measurement.decode_tokens as f64 / (measurement.decode_ms / 1000.0)
        } else {
            0.0
        };
        match &measurement.reading {
            Some(reading) => {
                row.energy_total_j = reading.energy.total().value() as f64;
                row.energy_cpu_j = reading.energy.cpu.value() as f64;
                row.energy_gpu_j = reading.energy.gpu.value() as f64;
                row.energy_gpu_sram_j = reading.energy.gpu_sram.value() as f64;
                row.energy_ane_j = reading.energy.ane.value() as f64;
                row.energy_ram_j = reading.energy.ram.value() as f64;
                row.avg_watts_total = reading.average_power.total().value() as f64;
                row.avg_watts_cpu = reading.average_power.cpu.value() as f64;
                row.avg_watts_gpu = reading.average_power.gpu.value() as f64;
                row.avg_watts_gpu_sram = reading.average_power.gpu_sram.value() as f64;
                row.avg_watts_ane = reading.average_power.ane.value() as f64;
                row.avg_watts_ram = reading.average_power.ram.value() as f64;
            },
            None => {
                row.status = "ok_no_energy".to_string();
            },
        }
        row
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
