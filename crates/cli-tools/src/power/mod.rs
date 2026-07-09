mod artifacts;
mod download;
mod local;
mod report;
mod workload;

use std::{
    any::Any,
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use backend_uzu::{backends::metal::Metal, engine::Engine, summarize_header};
use keisoku::{Device as MetricsDevice, PowerMeter};
use shoji::types::model::Model;
use tokio::{runtime::Handle as TokioHandle, time::sleep};
use uzu::engine::{Engine as UzuEngine, EngineConfig};

use self::{
    download::{Downloader, ModelFiles, weights_size},
    local::LocalArtifact,
    report::{DeviceInfo, Report, Row, Source},
};

const FAILURE_COOLDOWN_MULTIPLIER: u32 = 2;
const DEFAULT_REGISTRY_CACHE_DIR: &str = "uzu-power-consumption";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceMode {
    Registry,
    Local,
}

impl From<SourceMode> for Source {
    fn from(value: SourceMode) -> Self {
        match value {
            SourceMode::Registry => Self::Registry,
            SourceMode::Local => Self::Local,
        }
    }
}

pub struct Options {
    pub source: SourceMode,
    pub storage: Option<PathBuf>,
    pub output: PathBuf,
    pub model_ids: Vec<String>,
    pub prefill: Vec<usize>,
    pub generate: Vec<usize>,
    pub repetitions: usize,
    pub memory_fraction: f64,
    pub cooldown_secs: u64,
    pub weight_seed: u64,
}

/// Outcome of measuring a single model. Only `Faulted` warrants a longer cooldown
/// before the next model, giving the GPU command queue time to recover.
enum Outcome {
    Completed,
    Skipped,
    Faulted,
}

enum BenchmarkTarget {
    Registry(Model),
    Local(LocalArtifact),
}

struct ResolvedTarget {
    source: Source,
    id: String,
    files: ModelFiles,
}

struct Session {
    downloader: Option<Downloader>,
    meter: PowerMeter,
    report: Report,
    device_info: DeviceInfo,
    options: Options,
    input_tokens: Vec<u64>,
}

impl Session {
    async fn new(
        tokio: TokioHandle,
        options: Options,
    ) -> Result<Self> {
        let downloader = if options.source == SourceMode::Registry {
            let storage_base = registry_storage_base(options.storage.clone()).await?;
            Some(Downloader::new(tokio, Some(storage_base)).await?)
        } else {
            None
        };

        Ok(Self {
            downloader,
            meter: PowerMeter::new(),
            report: Report::new(&options.output)?,
            device_info: collect_device_info(),
            options,
            input_tokens: Vec::new(),
        })
    }

    async fn run(
        &mut self,
        targets: &[BenchmarkTarget],
    ) -> Result<()> {
        let cooldown = Duration::from_secs(self.options.cooldown_secs);
        let mut previous_faulted = false;
        for target in targets {
            let pause = if previous_faulted {
                cooldown * FAILURE_COOLDOWN_MULTIPLIER
            } else {
                cooldown
            };
            if !pause.is_zero() {
                sleep(pause).await;
            }

            let id = target_id(target);
            eprintln!("=> {id}");
            let outcome = self.measure_target(target).await;
            previous_faulted = matches!(outcome, Outcome::Faulted);
        }
        Ok(())
    }

    async fn measure_target(
        &mut self,
        target: &BenchmarkTarget,
    ) -> Outcome {
        if let Some(size) = estimated_memory_bytes(target)
            && self.should_skip_for_memory(size)
        {
            eprintln!("  skipped: model too large ({size} bytes)");
            return Outcome::Skipped;
        }

        let resolved = match self.resolve_target(target).await {
            Ok(resolved) => resolved,
            Err(error) => {
                eprintln!("  prepare failed: {error:#}");
                return Outcome::Faulted;
            },
        };

        let header_summary = match summarize_header(&resolved.files.header_path) {
            Ok(summary) => summary,
            Err(error) => {
                eprintln!("  prepare failed: {error:#}");
                return Outcome::Faulted;
            },
        };
        if self.should_skip_for_memory(header_summary.logical_payload_bytes) {
            eprintln!(
                "  skipped: model too large ({} bytes)",
                header_summary.logical_payload_bytes
            );
            return Outcome::Skipped;
        }

        let sweep = catch_unwind(AssertUnwindSafe(|| self.sweep_target(&resolved)));
        match sweep {
            Ok(Ok(())) => Outcome::Completed,
            Ok(Err(error)) => {
                eprintln!("  run failed: {error:#}");
                Outcome::Faulted
            },
            Err(payload) => {
                eprintln!("  panicked during model run: {}", panic_message(payload.as_ref()));
                Outcome::Faulted
            },
        }
    }

    async fn resolve_target(
        &self,
        target: &BenchmarkTarget,
    ) -> Result<ResolvedTarget> {
        match target {
            BenchmarkTarget::Registry(model) => {
                let downloader = self.downloader.as_ref().context("registry mode requires a downloader")?;
                let files = downloader.fetch(model).await?;
                Ok(ResolvedTarget {
                    source: Source::Registry,
                    id: model.identifier.clone(),
                    files,
                })
            },
            BenchmarkTarget::Local(artifact) => Ok(ResolvedTarget {
                source: Source::Local,
                id: artifact.id.clone(),
                files: ModelFiles {
                    config_path: artifact.config_path.clone(),
                    header_path: artifact.header_path.clone(),
                },
            }),
        }
    }

    /// Runs the prefill x generate sweep for one model, writing each measured repetition
    /// immediately. Aborts the remaining sweep on the first workload error so a faulted GPU
    /// queue is not hammered with follow-up submissions.
    fn sweep_target(
        &mut self,
        target: &ResolvedTarget,
    ) -> Result<()> {
        let engine = Engine::<Metal>::new().map_err(|error| anyhow!("engine init: {error}"))?;
        let language_model = engine
            .load_language_model_random(&target.files.config_path, &target.files.header_path, self.options.weight_seed)
            .map_err(|error| anyhow!("load model: {error}"))?;

        let context_limit = language_model.recommended_context_length();
        for &prefill in &self.options.prefill {
            for &generate in &self.options.generate {
                if let Some(limit) = context_limit
                    && prefill + generate > limit
                {
                    continue;
                }

                if let Err(error) = workload::run(&language_model, &mut self.meter, &mut self.input_tokens, prefill, 1)
                {
                    eprintln!("  warmup failed: {error:#}");
                }

                for _ in 0..self.options.repetitions {
                    let measurement =
                        workload::run(&language_model, &mut self.meter, &mut self.input_tokens, prefill, generate)?;
                    if let Some(row) = Row::measured(
                        &self.device_info,
                        target.source,
                        &target.id,
                        prefill,
                        generate,
                        &measurement,
                    ) {
                        self.report.write(&row)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn memory_limit(&self) -> u64 {
        memory_limit_bytes(self.device_info.ram_total_bytes, self.options.memory_fraction)
    }

    fn should_skip_for_memory(
        &self,
        size_bytes: u64,
    ) -> bool {
        exceeds_memory_limit(size_bytes, self.memory_limit())
    }
}

pub async fn run(
    tokio: TokioHandle,
    options: Options,
) -> Result<()> {
    validate_options(&options)?;

    let output = options.output.clone();
    let targets = prepare_targets(&options).await?;
    eprintln!("Benchmarking {} models", targets.len());

    let mut session = Session::new(tokio, options).await?;
    session.run(&targets).await?;

    eprintln!("Wrote {}", output.display());
    Ok(())
}

fn validate_options(options: &Options) -> Result<()> {
    if options.source == SourceMode::Local && options.storage.is_none() {
        bail!("--storage is required when --source local");
    }
    Ok(())
}

async fn prepare_targets(options: &Options) -> Result<Vec<BenchmarkTarget>> {
    match options.source {
        SourceMode::Registry => prepare_registry_targets(options).await,
        SourceMode::Local => prepare_local_targets(options),
    }
}

async fn prepare_registry_targets(options: &Options) -> Result<Vec<BenchmarkTarget>> {
    let engine_config = EngineConfig {
        mirai_api_key: None,
        ..EngineConfig::default()
    };
    let uzu_engine = UzuEngine::new(engine_config).await?;
    let mut models = uzu_engine.models().await?;
    models.retain(|model| model.is_downloadable() && model.is_chat_capable());
    models.retain(|model| matches_model_ids(&options.model_ids, &model.identifier));
    Ok(models.into_iter().map(BenchmarkTarget::Registry).collect())
}

fn prepare_local_targets(options: &Options) -> Result<Vec<BenchmarkTarget>> {
    let storage_base = options.storage.clone().context("--storage is required for local mode")?;
    let artifacts = local::discover(&storage_base, &options.model_ids)?;
    Ok(artifacts.into_iter().map(BenchmarkTarget::Local).collect())
}

async fn registry_storage_base(storage: Option<PathBuf>) -> Result<PathBuf> {
    let base = storage.unwrap_or_else(default_registry_storage_base);
    tokio::fs::create_dir_all(&base)
        .await
        .with_context(|| format!("create {}", base.display()))?;
    Ok(base)
}

pub fn default_registry_storage_base() -> PathBuf {
    std::env::temp_dir().join(DEFAULT_REGISTRY_CACHE_DIR)
}

pub fn effective_storage_base(storage: Option<PathBuf>) -> PathBuf {
    storage.unwrap_or_else(default_registry_storage_base)
}

pub fn cache_models_path(storage_base: &Path) -> PathBuf {
    artifacts::cache_models_path(storage_base)
}

fn estimated_memory_bytes(target: &BenchmarkTarget) -> Option<u64> {
    match target {
        BenchmarkTarget::Registry(model) => registry_memory_bytes(model),
        BenchmarkTarget::Local(artifact) => Some(artifact.header_summary.logical_payload_bytes),
    }
}

fn registry_memory_bytes(model: &Model) -> Option<u64> {
    weights_size(model)
        .map(|size| size as u64)
        .or_else(|| model.properties.as_ref().map(|properties| properties.size as u64))
}

pub fn memory_limit_bytes(ram_total_bytes: u64, memory_fraction: f64) -> u64 {
    (ram_total_bytes as f64 * memory_fraction) as u64
}

pub fn exceeds_memory_limit(size_bytes: u64, limit_bytes: u64) -> bool {
    size_bytes > limit_bytes
}

fn matches_model_ids(selected: &[String], id: &str) -> bool {
    selected.is_empty() || selected.iter().any(|candidate| candidate == id)
}

fn target_id(target: &BenchmarkTarget) -> &str {
    match target {
        BenchmarkTarget::Registry(model) => &model.identifier,
        BenchmarkTarget::Local(artifact) => &artifact.id,
    }
}

fn collect_device_info() -> DeviceInfo {
    let mut device = MetricsDevice::new();
    let ram_total_bytes = device.memory().map(|memory| memory.ram_total.value()).unwrap_or_default();
    DeviceInfo {
        os: device.os_version(),
        chip: device.chip(),
        ram_total_bytes,
        gpu_cores: device.gpu_cores(),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_mode_requires_storage_option() {
        let options = Options {
            source: SourceMode::Local,
            storage: None,
            output: PathBuf::from("out.csv"),
            model_ids: Vec::new(),
            prefill: vec![128],
            generate: vec![32],
            repetitions: 1,
            memory_fraction: 0.75,
            cooldown_secs: 0,
            weight_seed: 0,
        };
        let error = validate_options(&options).unwrap_err();
        assert!(error.to_string().contains("--storage"));
    }

    #[test]
    fn matches_model_ids_exactly() {
        assert!(matches_model_ids(&["alpha".to_string()], "alpha"));
        assert!(!matches_model_ids(&["alpha".to_string()], "alpha-extra"));
        assert!(matches_model_ids(&[], "anything"));
    }

    #[test]
    fn memory_limit_scales_with_ram_fraction() {
        assert_eq!(memory_limit_bytes(8_000_000_000, 0.75), 6_000_000_000);
    }

    #[test]
    fn exceeds_memory_limit_when_size_above_cap() {
        let limit = memory_limit_bytes(8_000_000_000, 0.75);
        assert!(exceeds_memory_limit(limit + 1, limit));
        assert!(!exceeds_memory_limit(limit, limit));
    }

    #[test]
    fn default_registry_storage_base_uses_temp() {
        let base = default_registry_storage_base();
        assert!(base.starts_with(std::env::temp_dir()));
        assert!(base.ends_with(DEFAULT_REGISTRY_CACHE_DIR));
    }
}
