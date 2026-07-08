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
use keisoku::{Chip, GpuCores, Os, RamTotal, Select, Static};
use shoji::types::model::Model;
use tokio::runtime::Handle as TokioHandle;
use uzu::{
    device::Device,
    engine::{Engine as UzuEngine, EngineConfig},
    storage::{Config as StorageConfig, DownloadContents, Storage},
};

use crate::{
    download::ModelFiles,
    report::{DeviceInfo, ModelMeta, Report, Row},
};

const SEED: u64 = 0;
const MEMORY_FRACTION: f64 = 0.75;
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(180);

#[derive(Parser)]
#[command(name = "power-consumption", about = "Measure per-model power/energy consumption across workloads")]
struct Args {
    /// Output CSV path.
    #[arg(long, default_value = "power_consumption.csv")]
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
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let tokio_handle = TokioHandle::current();

    let constants = Static::<Select![Os, Chip, RamTotal, GpuCores]>::new().into_sample();
    let device_info = DeviceInfo {
        os: constants.get::<Os>().clone(),
        chip: constants.get::<Chip>().clone(),
        ram_total_bytes: constants.get::<RamTotal>().value(),
        gpu_cores: *constants.get::<GpuCores>(),
    };

    let mut meter = workload::Meter::new();
    let mut report = Report::new(&args.output)?;
    let temp_dir = std::env::temp_dir().join("uzu-power-consumption");
    std::fs::create_dir_all(&temp_dir)?;

    let engine_config = EngineConfig {
        mirai_api_key: None,
        ..EngineConfig::default()
    };
    let uzu_engine = UzuEngine::new(engine_config).await?;
    let mut models = uzu_engine.models().await?;
    models.retain(|model| model.is_downloadable() && model.is_chat_capable());
    if let Some(filter) = &args.models {
        models.retain(|model| model.identifier.contains(filter));
    }
    eprintln!("Benchmarking {} models", models.len());

    let storage_device = Device::new().map_err(|error| anyhow!("device: {error}"))?;
    let storage_config = StorageConfig::new(storage_device, Some(temp_dir.clone()), "mirai".to_string())
        .with_download_contents(DownloadContents::CONFIG);
    let storage = Storage::new(tokio_handle, storage_config).await.map_err(|error| anyhow!("storage: {error}"))?;

    for model in &models {
        eprintln!("=> {}", model.identifier);
        match run_registry_model(&storage, &mut meter, &temp_dir, model, &device_info, &args).await {
            Ok(rows) => {
                for row in &rows {
                    report.write(row)?;
                }
                report.flush()?;
            },
            Err(error) => eprintln!("  error: {error}"),
        }
    }

    eprintln!("Wrote {}", args.output.display());
    Ok(())
}

async fn run_registry_model(
    storage: &Storage,
    meter: &mut workload::Meter,
    temp_dir: &Path,
    model: &Model,
    device_info: &DeviceInfo,
    args: &Args,
) -> Result<Vec<Row>> {
    if let Some(size) =
        download::weights_size(model).or_else(|| model.properties.as_ref().map(|properties| properties.size))
        && size as f64 > device_info.ram_total_bytes as f64 * MEMORY_FRACTION
    {
        eprintln!("  skipped: model too large");
        return Ok(vec![]);
    }

    let files = download::fetch(storage, model, temp_dir, DOWNLOAD_TIMEOUT).await?;
    let meta = ModelMeta::from_model(model);

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_model_files(&files, &meta, meter, device_info, args)
    }));
    match outcome {
        Ok(Ok(rows)) => Ok(rows),
        Ok(Err(error)) => {
            eprintln!("  run failed: {error}");
            Ok(vec![])
        },
        Err(_) => {
            eprintln!("  panicked during model run");
            Ok(vec![])
        },
    }
}

fn run_model_files(
    files: &ModelFiles,
    meta: &ModelMeta,
    meter: &mut workload::Meter,
    device_info: &DeviceInfo,
    args: &Args,
) -> Result<Vec<Row>> {
    let engine = Engine::<Metal>::new().map_err(|error| anyhow!("engine init: {error}"))?;
    let model = engine
        .load_language_model_random(&files.config_path, &files.header_path, SEED)
        .map_err(|error| anyhow!("load model: {error}"))?;

    let context_limit = model.recommended_context_length();
    let mut rows = Vec::new();
    for &prefill in &args.prefill {
        for &generate in &args.generate {
            if let Some(limit) = context_limit
                && prefill + generate > limit
            {
                continue;
            }

            let _ = workload::run(&model, meter, prefill, 1);

            for _ in 0..args.repetitions {
                match workload::run(&model, meter, prefill, generate) {
                    Ok(measurement) => {
                        if let Some(row) = Row::measured(device_info, meta, prefill, generate, &measurement) {
                            rows.push(row);
                        }
                    },
                    Err(error) => eprintln!("  run failed: {error}"),
                }
            }
        }
    }
    Ok(rows)
}
