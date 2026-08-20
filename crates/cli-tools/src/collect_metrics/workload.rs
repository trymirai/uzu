use std::time::Instant;

use anyhow::{Result, anyhow};
use backend_uzu::{
    backends::metal::Metal,
    engine::language_model::{
        LanguageModel,
        stream::{LanguageModelStreamOptions, SamplingMethod},
    },
};
use keisoku::PowerMeter;

pub struct Reading {
    pub energy_total_j: f64,
    pub total_watts: f64,
    pub cpu_watts: f64,
    pub gpu_watts: f64,
    pub ram_watts: f64,
    pub dram_read_bytes: u64,
    pub dram_write_bytes: u64,
    pub dram_read_gbps: f64,
    pub dram_write_gbps: f64,
}

pub struct Measurement {
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub decode_tokens: usize,
    pub reading: Option<Reading>,
}

pub fn run(
    model: &LanguageModel<Metal>,
    meter: &mut PowerMeter,
    input: &mut Vec<u64>,
    prefill: usize,
    generate: usize,
) -> Result<Measurement> {
    input.clear();
    input.extend((0..prefill).map(|index| (index % 7) as u64));

    let mut state = model
        .create_empty_state(model.recommended_context_length())
        .map_err(|error| anyhow!("create_empty_state: {error}"))?;
    let options = LanguageModelStreamOptions {
        sampling_method: SamplingMethod::Greedy,
        grammar: None,
        speculator: None,
    };

    meter.start();
    let start = Instant::now();

    let mut stream = model.stream(input, &mut state, options).map_err(|error| anyhow!("stream: {error}"))?;

    match stream.next() {
        Some(Ok(_)) => {},
        Some(Err(error)) => return Err(anyhow!("prefill: {error}")),
        None => {},
    }
    let after_prefill = Instant::now();

    let mut decode_tokens = 0usize;
    for _ in 1..generate {
        match stream.next() {
            Some(Ok(_)) => decode_tokens += 1,
            Some(Err(error)) => return Err(anyhow!("decode: {error}")),
            None => break,
        }
    }
    let end = Instant::now();
    drop(stream);

    let reading = meter.stop().map(|power| Reading {
        energy_total_j: power.energy.value() as f64,
        total_watts: power.total.value() as f64,
        cpu_watts: power.cpu.map(|watts| watts.value() as f64).unwrap_or(0.0),
        gpu_watts: power.gpu.map(|watts| watts.value() as f64).unwrap_or(0.0),
        ram_watts: power.ram.map(|watts| watts.value() as f64).unwrap_or(0.0),
        dram_read_bytes: power.dram_read_bytes.unwrap_or(0),
        dram_write_bytes: power.dram_write_bytes.unwrap_or(0),
        dram_read_gbps: power.dram_read_gbps.map(|gbps| gbps as f64).unwrap_or(0.0),
        dram_write_gbps: power.dram_write_gbps.map(|gbps| gbps as f64).unwrap_or(0.0),
    });

    Ok(Measurement {
        prefill_ms: (after_prefill - start).as_secs_f64() * 1000.0,
        decode_ms: (end - after_prefill).as_secs_f64() * 1000.0,
        decode_tokens,
        reading,
    })
}
