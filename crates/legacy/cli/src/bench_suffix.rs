use std::{fs::OpenOptions, io::BufWriter, path::Path, time::Duration};

use anyhow::Result;
use backend_uzu::{backends::metal::Metal, engine::Engine};
use serde::Serialize;

#[derive(Serialize)]
struct SuffixResult {
    suffix_length: u32,
    seconds: Box<[f64]>,
    mean_seconds: f64,
    tokens_per_second: f64,
    sampled_tokens: Box<[u32]>,
}

#[derive(Serialize)]
struct Report {
    model_path: String,
    prefix_length: u32,
    warmup_runs: u32,
    measured_runs: u32,
    peak_memory_bytes: Option<usize>,
    suffixes: Box<[SuffixResult]>,
}

pub fn run(
    model_path: String,
    output_path: String,
    prefix_length: u32,
    suffix_lengths: &[u32],
    warmup_runs: u32,
    measured_runs: u32,
) -> Result<()> {
    let engine = Engine::<Metal>::new()?;
    let model = engine.load_language_model(Path::new(&model_path))?;
    let benchmarks = model.benchmark_suffix_forwards(prefix_length, suffix_lengths, warmup_runs, measured_runs)?;
    let suffixes = benchmarks
        .into_iter()
        .map(|benchmark| {
            let seconds = benchmark.durations.iter().map(Duration::as_secs_f64).collect::<Box<[_]>>();
            let mean_seconds = seconds.iter().sum::<f64>() / seconds.len() as f64;
            SuffixResult {
                suffix_length: benchmark.suffix_length,
                seconds,
                mean_seconds,
                tokens_per_second: benchmark.suffix_length as f64 / mean_seconds,
                sampled_tokens: benchmark.sampled_tokens.clone(),
            }
        })
        .collect::<Box<[_]>>();
    let report = Report {
        model_path,
        prefix_length,
        warmup_runs,
        measured_runs,
        peak_memory_bytes: engine.peak_memory_usage(),
        suffixes,
    };

    let output = OpenOptions::new().write(true).create_new(true).open(output_path)?;
    serde_json::to_writer_pretty(BufWriter::new(output), &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
