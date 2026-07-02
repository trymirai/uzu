use std::time::{Duration, Instant};

use crate::backends::{
    common::Encoder,
    metal::{Metal, MetalContext},
};

const WARMUP_BATCHES: usize = 2;
const BATCH_SIZE: u64 = 64;
const SAMPLE_INTERVAL: Duration = Duration::from_millis(100);

pub(crate) const OUTPUT_BUFFER_ROTATION: usize = 4;

#[derive(Clone, Copy)]
pub struct PowerSample {
    pub elapsed_milliseconds: u64,
    pub cpu_watts: f64,
    pub gpu_watts: f64,
    pub gpu_sram_watts: f64,
    pub ram_watts: f64,
    pub dram_read_gigabytes_per_second: f64,
    pub dram_write_gigabytes_per_second: f64,
}

impl PowerSample {
    const CSV_COLUMNS: &str = "elapsed_milliseconds,cpu_watts,gpu_watts,gpu_sram_watts,ram_watts,dram_read_gigabytes_per_second,dram_write_gigabytes_per_second";

    fn csv_row(&self) -> String {
        format!(
            "{},{},{},{},{},{},{}",
            self.elapsed_milliseconds,
            self.cpu_watts,
            self.gpu_watts,
            self.gpu_sram_watts,
            self.ram_watts,
            self.dram_read_gigabytes_per_second,
            self.dram_write_gigabytes_per_second,
        )
    }
}

pub struct Measurement {
    pub iterations: u64,
    pub gpu_time_per_iteration_microseconds: f64,
    pub power_samples: Vec<PowerSample>,
}

impl Measurement {
    pub fn measure(
        context: &MetalContext,
        window: Duration,
        mut encode_once: impl FnMut(&mut Encoder<Metal>),
    ) -> Self {
        for _ in 0..WARMUP_BATCHES {
            run_batch(context, &mut encode_once);
        }

        let recorder = keisoku::start(keisoku::Config {
            interval: SAMPLE_INTERVAL,
        });
        let start = Instant::now();
        let mut iterations = 0u64;
        let mut gpu_time_total = Duration::ZERO;
        while start.elapsed() < window {
            gpu_time_total += run_batch(context, &mut encode_once);
            iterations += BATCH_SIZE;
        }
        let power_samples = collect_power_samples(&recorder.stop());

        Self {
            iterations,
            gpu_time_per_iteration_microseconds: gpu_time_total.as_nanos() as f64 / 1000.0 / iterations as f64,
            power_samples,
        }
    }

    pub fn csv_header() -> String {
        format!("iterations,gpu_time_per_iteration_microseconds,sample_index,{}", PowerSample::CSV_COLUMNS)
    }

    pub fn csv_rows(
        &self,
        parameter_fields: &str,
    ) -> Vec<String> {
        self.power_samples
            .iter()
            .enumerate()
            .map(|(sample_index, sample)| {
                format!(
                    "{parameter_fields},{},{},{sample_index},{}",
                    self.iterations,
                    self.gpu_time_per_iteration_microseconds,
                    sample.csv_row(),
                )
            })
            .collect()
    }
}

fn run_batch(
    context: &MetalContext,
    encode_once: &mut impl FnMut(&mut Encoder<Metal>),
) -> Duration {
    let mut encoder = Encoder::<Metal>::new(context).unwrap();
    for _ in 0..BATCH_SIZE {
        encode_once(&mut encoder);
    }
    encoder.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time()
}

fn collect_power_samples(session: &keisoku::Session) -> Vec<PowerSample> {
    session
        .snapshots
        .iter()
        .skip(1)
        .filter_map(|snapshot| {
            let power = snapshot.power.as_ref()?;
            let (dram_read_gigabytes_per_second, dram_write_gigabytes_per_second) = snapshot
                .bandwidth
                .as_ref()
                .map(|bandwidth| (bandwidth.dram_read.value() as f64, bandwidth.dram_write.value() as f64))
                .unwrap_or((0.0, 0.0));
            Some(PowerSample {
                elapsed_milliseconds: snapshot.elapsed.value(),
                cpu_watts: power.cpu.value() as f64,
                gpu_watts: power.gpu.value() as f64,
                gpu_sram_watts: power.gpu_sram.value() as f64,
                ram_watts: power.ram.value() as f64,
                dram_read_gigabytes_per_second,
                dram_write_gigabytes_per_second,
            })
        })
        .collect()
}
