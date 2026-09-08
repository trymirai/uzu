use std::{
    any::Any,
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use tokio::time::sleep;
use uzu::engine::{Engine as UzuEngine, EngineConfig};
use uzu_engine::{backends::metal::Metal, engine::Engine};

use super::{
    benchmark_target::BenchmarkTarget, device_info::DeviceInfo, downloader::Downloader, local_artifact::LocalArtifact,
    measurement::Measurement, options::Options, report::Report, resolved_target::ResolvedTarget, row::Row,
};
use crate::types::SourceMode;

const WEIGHT_SEED: u64 = 0;
const MEMORY_FRACTION: f64 = 0.75;
const COOLDOWN: Duration = Duration::from_secs(3);
const FAILURE_COOLDOWN_MULTIPLIER: u32 = 2;
const REGISTRY_CACHE_DIR: &str = "uzu-collect-metrics";

/// Owns the sweep: resolves every target, measures it, and appends rows to the report.
pub struct Session {
    downloader: Option<Downloader>,
    report: Report,
    device_info: DeviceInfo,
    options: Options,
    input_tokens: Vec<u64>,
}

impl Session {
    pub async fn run(options: Options) -> Result<()> {
        if options.source == SourceMode::Local && options.storage.is_none() {
            bail!("--storage is required when --source local");
        }

        let output = options.output.clone();
        let targets = Self::targets(&options).await?;
        eprintln!("Benchmarking {} models", targets.len());

        let mut session = Self::new(options).await?;
        session.measure_all(&targets).await;

        eprintln!("Wrote {}", output.display());
        Ok(())
    }

    async fn new(options: Options) -> Result<Self> {
        let downloader = if options.source == SourceMode::Registry {
            Some(Downloader::new(Some(registry_storage_base(options.storage.clone()).await?)).await?)
        } else {
            None
        };

        Ok(Self {
            downloader,
            report: Report::new(&options.output)?,
            device_info: DeviceInfo::collect()?,
            options,
            input_tokens: Vec::new(),
        })
    }

    async fn targets(options: &Options) -> Result<Vec<BenchmarkTarget>> {
        match options.source {
            SourceMode::Registry => {
                let engine = UzuEngine::new(EngineConfig::default()).await?;
                let mut models = engine.models().await?;
                models.retain(|model| {
                    model.is_downloadable() && model.is_chat_capable() && options.selects(&model.identifier)
                });
                Ok(models.into_iter().map(BenchmarkTarget::Registry).collect())
            },
            SourceMode::Local => {
                let storage = options.storage.as_deref().context("--storage is required for local mode")?;
                Ok(LocalArtifact::discover(storage, options)?.into_iter().map(BenchmarkTarget::Local).collect())
            },
        }
    }

    /// A faulted model gets a longer cooldown before the next one, to let the GPU settle.
    async fn measure_all(
        &mut self,
        targets: &[BenchmarkTarget],
    ) {
        let mut previous_faulted = false;
        for target in targets {
            sleep(if previous_faulted {
                COOLDOWN * FAILURE_COOLDOWN_MULTIPLIER
            } else {
                COOLDOWN
            })
            .await;

            eprintln!("=> {}", target.id());
            previous_faulted = self.measure_target(target).await;
        }
    }

    /// Returns whether the target faulted, i.e. failed to prepare, run, or survive its sweep.
    async fn measure_target(
        &mut self,
        target: &BenchmarkTarget,
    ) -> bool {
        if let Some(size) = target.estimated_memory_bytes()
            && size > (self.device_info.ram_total_bytes as f64 * MEMORY_FRACTION) as u64
        {
            eprintln!("  skipped: model too large ({size} bytes)");
            return false;
        }

        let resolved = match self.resolve(target).await {
            Ok(resolved) => resolved,
            Err(error) => {
                eprintln!("  prepare failed: {error:#}");
                return true;
            },
        };

        match catch_unwind(AssertUnwindSafe(|| self.sweep(&resolved))) {
            Ok(Ok(())) => false,
            Ok(Err(error)) => {
                eprintln!("  run failed: {error:#}");
                true
            },
            Err(payload) => {
                eprintln!("  panicked during model run: {}", panic_message(payload.as_ref()));
                true
            },
        }
    }

    async fn resolve(
        &self,
        target: &BenchmarkTarget,
    ) -> Result<ResolvedTarget> {
        if let Some(resolved) = target.resolved_locally() {
            return Ok(resolved);
        }
        let BenchmarkTarget::Registry(model) = target else {
            unreachable!("a non-local target is a registry target");
        };
        self.downloader.as_ref().context("registry mode requires a downloader")?.fetch(model).await
    }

    fn sweep(
        &mut self,
        target: &ResolvedTarget,
    ) -> Result<()> {
        let engine = Engine::<Metal>::new().map_err(|error| anyhow!("engine init: {error}"))?;
        let model = engine
            .load_language_model_random(&target.model_dir, &target.header_path, WEIGHT_SEED)
            .map_err(|error| anyhow!("load model: {error}"))?;

        let context_limit = model.recommended_context_length();
        for &prefill in &self.options.prefill {
            for &generate in &self.options.generate {
                if let Some(limit) = context_limit
                    && prefill + generate > limit as usize
                {
                    continue;
                }

                if let Err(error) = Measurement::capture(&model, &mut self.input_tokens, prefill, 1) {
                    eprintln!("  warmup failed: {error:#}");
                }

                for _ in 0..self.options.iterations {
                    let measurement = Measurement::capture(&model, &mut self.input_tokens, prefill, generate)?;
                    if let Some(row) = Row::measured(&self.device_info, target, prefill, generate, &measurement) {
                        self.report.write(&row)?;
                    }
                }
            }
        }
        Ok(())
    }
}

async fn registry_storage_base(storage: Option<PathBuf>) -> Result<PathBuf> {
    let base = storage.unwrap_or_else(|| std::env::temp_dir().join(REGISTRY_CACHE_DIR));
    tokio::fs::create_dir_all(&base).await.with_context(|| format!("create {}", base.display()))?;
    Ok(base)
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic".to_string()
    }
}
