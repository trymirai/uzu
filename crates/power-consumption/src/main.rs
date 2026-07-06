mod download;
mod report;
mod workload;

use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Result, anyhow};
use backend_uzu::{backends::metal::Metal, engine::Engine};
use clap::Parser;
use keisoku::{Collector, EnergyMeter};
use shoji::types::model::Model;
use uzu::{
    device::Device,
    engine::{Engine as UzuEngine, EngineConfig},
    storage::{Config as StorageConfig, DownloadContents, Storage},
};

use crate::{
    download::ModelFiles,
    report::{DeviceInfo, ModelMeta, Report, Row},
};

#[derive(Parser)]
#[command(name = "power-consumption", about = "Measure per-model power/energy consumption across workloads")]
struct Args {
    /// Output CSV path.
    #[arg(long, default_value = "power_consumption_report.csv")]
    output: PathBuf,
    /// Only benchmark models whose identifier contains this substring.
    #[arg(long)]
    models: Option<String>,
    /// Prefill (prompt) token counts to sweep.
    #[arg(long, value_delimiter = ',', default_value = "128,512,2048")]
    prefill: Vec<usize>,
    /// Generation lengths to sweep.
    #[arg(long, value_delimiter = ',', default_value = "32,128")]
    generate: Vec<usize>,
    /// Measured repetitions per (prefill, generate) combination.
    #[arg(long, default_value_t = 3)]
    repetitions: usize,
    /// Seed for the random weights.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Include models that are not chat-capable (classification, TTS, ...).
    #[arg(long)]
    include_non_chat: bool,
    /// Fraction of total RAM a model may occupy before it is skipped as too large.
    #[arg(long, default_value_t = 0.75)]
    memory_fraction: f64,
    /// Benchmark a single on-disk model directory (config.json + model.safetensors)
    /// instead of enumerating and downloading from the registry.
    #[arg(long)]
    local_model_path: Option<PathBuf>,
    /// Load the real weights from --local-model-path instead of random ones (calibration).
    #[arg(long)]
    real_weights: bool,
    /// Timeout (seconds) for downloading a single model's config.
    #[arg(long, default_value_t = 180)]
    download_timeout_secs: u64,
    /// Also print the CSV to stdout wrapped in BEGIN/END markers (for CI log capture).
    #[arg(long)]
    print_csv: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let runtime = tokio::runtime::Runtime::new()?;

    let device = Collector::new().device();
    let device_info = DeviceInfo {
        os: device.os.clone(),
        chip: device.chip.clone(),
        ram_total_bytes: device.ram_total.value(),
        gpu_cores: device.gpu_cores,
    };

    let meter = EnergyMeter::new();
    let mut report = Report::new(&args.output)?;
    let temp_dir = std::env::temp_dir().join("uzu-power-consumption");
    std::fs::create_dir_all(&temp_dir)?;

    if let Some(local) = args.local_model_path.clone() {
        run_local(&local, &meter, &device_info, &args, &mut report)?;
        report.flush()?;
        eprintln!("Wrote {}", args.output.display());
        emit_csv(&args)?;
        return Ok(());
    }

    let engine_config = EngineConfig {
        mirai_api_key: None,
        ..EngineConfig::default()
    };
    let uzu_engine = runtime.block_on(async { UzuEngine::new(engine_config).await })?;
    let mut models = runtime.block_on(uzu_engine.models())?;
    models.retain(|model| model.is_downloadable() && (args.include_non_chat || model.is_chat_capable()));
    if let Some(filter) = &args.models {
        models.retain(|model| model.identifier.contains(filter));
    }
    eprintln!("Benchmarking {} models", models.len());

    let storage = runtime.block_on(build_storage())?;

    for model in &models {
        eprintln!("=> {}", model.identifier);
        let rows = match run_registry_model(&runtime, &storage, &meter, &temp_dir, model, &device_info, &args) {
            Ok(rows) => rows,
            Err(error) => vec![Row::status(&device_info, &ModelMeta::from_model(model), "error", &error.to_string())],
        };
        for row in &rows {
            report.write(row)?;
        }
        report.flush()?;
    }

    eprintln!("Wrote {}", args.output.display());
    emit_csv(&args)?;
    Ok(())
}

fn emit_csv(args: &Args) -> Result<()> {
    if args.print_csv {
        let csv = std::fs::read_to_string(&args.output)?;
        println!("===CSV BEGIN===");
        print!("{csv}");
        if !csv.ends_with('\n') {
            println!();
        }
        println!("===CSV END===");
    }
    Ok(())
}

async fn build_storage() -> Result<Storage> {
    let device = Device::new().map_err(|error| anyhow!("device: {error}"))?;
    let config = StorageConfig::new(device, None, "mirai".to_string()).with_download_contents(DownloadContents::CONFIG);
    Storage::new(tokio::runtime::Handle::current(), config).await.map_err(|error| anyhow!("storage: {error}"))
}

fn run_local(
    local: &Path,
    meter: &EnergyMeter,
    device_info: &DeviceInfo,
    args: &Args,
    report: &mut Report,
) -> Result<()> {
    let files = ModelFiles {
        config_path: local.join("config.json"),
        header_path: local.join("model.safetensors"),
    };
    let id = local.file_name().map(|name| name.to_string_lossy().to_string()).unwrap_or_else(|| "local".to_string());
    let meta = ModelMeta {
        id: id.clone(),
        name: id,
        ..Default::default()
    }
    .with_config(&files.config_path);

    let rows = run_model_files(&files, Some(local), &meta, meter, device_info, args)
        .unwrap_or_else(|error| vec![Row::status(device_info, &meta, "run_failed", &error.to_string())]);
    for row in &rows {
        report.write(row)?;
    }
    Ok(())
}

fn run_registry_model(
    runtime: &tokio::runtime::Runtime,
    storage: &Storage,
    meter: &EnergyMeter,
    temp_dir: &Path,
    model: &Model,
    device_info: &DeviceInfo,
    args: &Args,
) -> Result<Vec<Row>> {
    let meta = ModelMeta::from_model(model);

    if let Some(size) = meta.size_bytes
        && size as f64 > device_info.ram_total_bytes as f64 * args.memory_fraction
    {
        return Ok(vec![Row::status(device_info, &meta, "skipped_too_large", "")]);
    }

    let files =
        runtime.block_on(download::fetch(storage, model, temp_dir, Duration::from_secs(args.download_timeout_secs)))?;
    let meta = meta.with_config(&files.config_path);

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_model_files(&files, None, &meta, meter, device_info, args)
    }));
    match outcome {
        Ok(Ok(rows)) => Ok(rows),
        Ok(Err(error)) => Ok(vec![Row::status(device_info, &meta, "run_failed", &error.to_string())]),
        Err(_) => Ok(vec![Row::status(device_info, &meta, "panic", "panicked during model run")]),
    }
}

fn run_model_files(
    files: &ModelFiles,
    real_weights_dir: Option<&Path>,
    meta: &ModelMeta,
    meter: &EnergyMeter,
    device_info: &DeviceInfo,
    args: &Args,
) -> Result<Vec<Row>> {
    let engine = Engine::<Metal>::new().map_err(|error| anyhow!("engine init: {error}"))?;
    let model = match real_weights_dir {
        Some(dir) if args.real_weights => {
            engine.load_language_model(dir).map_err(|error| anyhow!("load real model: {error}"))?
        },
        _ => engine
            .load_language_model_random(&files.config_path, &files.header_path, args.seed)
            .map_err(|error| anyhow!("load model: {error}"))?,
    };

    let context_limit = model.max_context_length();
    let mut rows = Vec::new();
    for &prefill in &args.prefill {
        for &generate in &args.generate {
            if let Some(limit) = context_limit
                && prefill + generate > limit
            {
                let mut row = Row::status(device_info, meta, "skipped_context", "prefill+generate exceeds context");
                row.prefill_tokens = prefill;
                row.generate_tokens = generate;
                rows.push(row);
                continue;
            }

            let _ = workload::run(&model, meter, prefill, 1);

            for repetition in 0..args.repetitions {
                match workload::run(&model, meter, prefill, generate) {
                    Ok(measurement) => {
                        rows.push(Row::measured(device_info, meta, prefill, generate, repetition, &measurement));
                    },
                    Err(error) => {
                        rows.push(Row::status(device_info, meta, "run_failed", &error.to_string()));
                    },
                }
            }
        }
    }
    Ok(rows)
}
