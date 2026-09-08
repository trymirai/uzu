use std::{fs::File, path::Path};

use anyhow::Result;

use super::row::Row;

const HEADER: [&str; 26] = [
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
    "dram_read_bytes",
    "dram_write_bytes",
    "dram_read_gbps",
    "dram_write_gbps",
];

/// CSV sink that flushes after every row so a long sweep survives an interrupt.
pub struct Report {
    writer: csv::Writer<File>,
}

impl Report {
    pub fn new(path: &Path) -> Result<Self> {
        let mut writer = csv::WriterBuilder::new().has_headers(false).from_path(path)?;
        writer.write_record(HEADER)?;
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
