use std::{net::IpAddr, path::PathBuf, sync::Arc};

use anyhow::{Context, Result, bail};
use rocket::{Config, config::LogLevel, routes};
use tokio::sync::Mutex;
use uzu::{
    engine::{Engine, EngineConfig},
    types::model::ModelSpecialization,
};

use uzu_engine::d1_omni::D1OmniPipeline;

use crate::{
    server::{
        ChatState, DecisionBackend, DecisionState, ServerState, SwapSlot,
        access_log::{AccessLog, AccessLogFairing},
        handle_chat_completions, handle_decision_model_load, handle_decision_model_unload, handle_decisions,
        handle_model_load, handle_model_status, handle_model_unload,
        handle_audio_capabilities, handle_audio_speech, handle_audio_transcriptions_json,
        handle_audio_transcriptions_multipart, handle_audio_translate, handle_live_asr_chunk,
        handle_live_asr_start, handle_live_asr_stop, handle_omni_capabilities, handle_web_ui,
        handle_models, handle_models_static, handle_raw_completions, handle_server_state, handle_system_one,
        loader::{LoadParams, clamp_context_tokens, load_model, prepare_downloads},
        log::{self, FileLogger, RequestLoggingFairing},
    },
};

pub async fn load_decision_backend(
    engine: &Engine,
    target: &str,
) -> Result<Option<(String, DecisionBackend)>> {
    let target_path = PathBuf::from(target);
    if D1OmniPipeline::is_d1_omni(&target_path) {
        tracing::info!("🧠 Loading d1-Omni System One pipeline from {}", target_path.display());
        let pipeline = D1OmniPipeline::load_metal(&target_path)
            .await
            .context("Failed to load d1-Omni pipeline")?;
        let name = target_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(target)
            .to_string();
        return Ok(Some((name, DecisionBackend::D1Omni(Arc::new(Mutex::new(pipeline))))));
    }

    let Some(resolved) = engine.model(target.to_string()).await? else {
        return Ok(None);
    };
    let is_legacy_decision = resolved.specializations.contains(&ModelSpecialization::Decision {});
    prepare_downloads(engine, &resolved).await?;
    let resolved = engine
        .model(target.to_string())
        .await?
        .unwrap_or(resolved);
    if let uzu::types::model::ModelAccessibility::OnDevice {
        source: uzu::types::model::ModelSource::Filesystem {
            path,
        },
    } = &resolved.accessibility
    {
        let local_path = std::path::Path::new(path);
        if D1OmniPipeline::is_d1_omni(local_path) {
            tracing::info!("🧠 Loading d1-Omni System One pipeline from {}", local_path.display());
            let pipeline = D1OmniPipeline::load_metal(local_path)
                .await
                .context("Failed to load d1-Omni pipeline")?;
            return Ok(Some((
                resolved.identifier.clone(),
                DecisionBackend::D1Omni(Arc::new(Mutex::new(pipeline))),
            )));
        }
    }
    if is_legacy_decision {
        let session = engine.decision(resolved.clone()).await.context("Failed to create decision session")?;
        return Ok(Some((
            resolved.identifier,
            DecisionBackend::Legacy(Arc::new(Mutex::new(session))),
        )));
    }
    Ok(None)
}

/// `<cache>/mirai/server`, resolved like every other server-side log location.
fn server_cache_dir() -> anyhow::Result<PathBuf> {
    let cache_dir = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .or_else(|| dirs::home_dir().map(|path| path.join(".cache")))
        .context("Failed to resolve cache directory")?;
    Ok(cache_dir.join("mirai").join("server"))
}

pub async fn run_server(
    model: Option<String>,
    models_dir: Option<String>,
    host: String,
    port: u16,
    prefix_cache: bool,
    request_log: bool,
    verbose_file_logs: bool,
    verbose: bool,
    seed: Option<i64>,
    max_context_tokens: Option<u32>,
) -> Result<()> {
    if verbose {
        uzu_engine::verbose::set_verbose(true);
    }
    let file_logger = if verbose_file_logs {
        Some(FileLogger::new(server_cache_dir()?.join("logs"))?)
    } else {
        None
    };
    log::init(
        if verbose { tracing::Level::DEBUG } else { tracing::Level::INFO },
        file_logger.clone(),
        tracing::Level::DEBUG,
    )?;

    let access_log: std::sync::Arc<AccessLog> = if request_log {
        let path = server_cache_dir()?.join("requests.jsonl");
        match AccessLog::open(path.clone()) {
            Ok(access_log) => {
                tracing::info!("📝 Request journal: {}", path.display());
                std::sync::Arc::new(access_log)
            },
            Err(error) => {
                tracing::warn!("📝 Request journal unavailable ({}): {error}", path.display());
                std::sync::Arc::new(AccessLog::disabled())
            },
        }
    } else {
        tracing::info!("📝 Request journal: disabled (--no-request-log)");
        std::sync::Arc::new(AccessLog::disabled())
    };

    let engine_config = EngineConfig::default();
    let engine = Arc::new(Engine::new(engine_config).await.context("Failed to create engine")?);

    let model_name = model.clone().unwrap_or_else(|| "(dynamic; idle)".to_string());
    let address: IpAddr = host.parse().with_context(|| format!("Invalid host: {host}"))?;
    let config = Config {
        port,
        address,
        log_level: LogLevel::Off,
        ..Config::default()
    };

    let rocket = rocket::custom(config)
        .manage(ServerState {
            model_name: model.clone(),
            access_log: std::sync::Arc::clone(&access_log),
            started_at: chrono::Utc::now(),
            uptime: std::time::Instant::now(),
        })
        .attach(RequestLoggingFairing)
        .attach(AccessLogFairing);

    let load_params = LoadParams {
        seed,
        max_context_tokens: clamp_context_tokens(max_context_tokens),
    };
    let mut decision_backend: Option<(String, DecisionBackend)> = None;
    let mut preloaded = None;
    if let Some(target) = model.as_deref() {
        if let Some(loaded_dec) = load_decision_backend(&engine, target).await? {
            decision_backend = Some(loaded_dec);
        } else {
            let loaded = load_model(&engine, target, &load_params).await?;
            preloaded = Some(loaded);
        }
    }

    let (rocket, title, details) = if let Some((dec_name, dec_backend)) = decision_backend {
        let state = DecisionState {
            engine: Arc::clone(&engine),
            backend: Arc::new(tokio::sync::RwLock::new(Some((dec_name, dec_backend)))),
            load_gate: Arc::new(Mutex::new(())),
        };
        let rocket = rocket
            .manage(state)
            .mount(
                "/",
                routes![
                    handle_web_ui,
                    handle_system_one,
                    handle_decisions,
                    handle_models_static,
                    handle_decision_model_load,
                    handle_decision_model_unload
                ],
            )
            .mount(
                "/v1",
                routes![
                    handle_system_one,
                    handle_decisions,
                    handle_models_static,
                    handle_decision_model_load,
                    handle_decision_model_unload
                ],
            );
        let details = concat!(
            "📝 Endpoints:\n",
            "   GET  /                    (System One Decision Workbench WebUI)\n",
            "   POST /v1/systemone        (or /v1/decisions, /systemone, /decisions)\n",
            "        Docs: https://github.com/trymirai/uzu#decision-models\n",
        );
        (rocket, "System One server", details.to_string())
    } else {
        // One slot per process; empty unless `--model` preloaded it. Models
        // load/swap at runtime through the admin API.
        let slot = SwapSlot::new();
        if let Some(loaded) = preloaded {
            slot.set_ready(loaded).await;
        }
        let state = ChatState {
            engine: Arc::clone(&engine),
            slot: Arc::new(slot),
            load_gate: Arc::new(Mutex::new(())),
            prefix_cache,
            seed,
            max_context_tokens: load_params.max_context_tokens,
            models_dir: models_dir.clone().map(PathBuf::from),
        };
        let rocket = rocket
            .manage(state)
            .mount(
                "/",
                routes![
                    handle_web_ui,
                    handle_chat_completions,
                    handle_audio_transcriptions_multipart,
                    handle_audio_transcriptions_json,
                    handle_audio_speech,
                    handle_audio_capabilities,
                    handle_omni_capabilities,
                    handle_audio_translate,
                    handle_live_asr_start,
                    handle_live_asr_chunk,
                    handle_live_asr_stop,
                    handle_models,
                    handle_server_state,
                    handle_model_status,
                    handle_model_load,
                    handle_model_unload,
                    handle_raw_completions
                ],
            )
            .mount(
                "/v1",
                routes![
                    handle_chat_completions,
                    handle_audio_transcriptions_multipart,
                    handle_audio_transcriptions_json,
                    handle_audio_speech,
                    handle_audio_capabilities,
                    handle_omni_capabilities,
                    handle_audio_translate,
                    handle_live_asr_start,
                    handle_live_asr_chunk,
                    handle_live_asr_stop,
                    handle_models,
                    handle_server_state,
                    handle_model_status,
                    handle_model_load,
                    handle_model_unload,
                    handle_raw_completions
                ],
            );
        let prefix_cache_enabled = if prefix_cache {
            "enabled"
        } else {
            "disabled"
        };
        let details = format!(
            concat!(
                "🗄️  Prefix cache: {prefix_cache_enabled}\n",
                "📦 Models dir: {models_dir}\n",
                "📝 Endpoints:\n",
                "   GET  /                      — Web UI workbench\n",
                "   POST /v1/chat/completions   (or /chat/completions; supports input_audio & modalities)\n",
                "   POST /v1/audio/transcriptions (multipart file or JSON audio_base64/audio_path)\n",
                "   POST /v1/audio/speech         (text -> 24 kHz WAV / PCM / SSE stream)\n",
                "   POST /v1/audio/translate      (multilingual speech/text translation)\n",
                "   POST /v1/audio/live-asr/{{start,chunk,stop}} (English causal streaming ASR)\n",
                "   GET  /v1/audio/capabilities   (LFM3-Audio capability & constraint manifest)\n",
                "   GET  /v1/state            (or /state) — server internals\n",
                "   GET  /v1/models/status    — resident model / slot state\n",
                "   POST /v1/models/load      — load a model: {{\"model\": \"<name>\"}}\n",
                "   POST /v1/models/unload    — free the resident model\n",
            ),
            prefix_cache_enabled = prefix_cache_enabled,
            models_dir = models_dir.as_deref().unwrap_or("(none; set --models-dir for GET /v1/models listings)"),
        );
        (rocket, "OpenAI-compatible server", details)
    };

    if let Some(ref file_logger) = file_logger {
        tracing::info!("🗃️  Logs will be written to file: {}", file_logger.file_path().display())
    }
    tracing::info!(
        concat!(
            "🚀 {title} v{version} for model: {model_name}\n",
            "🌐 Available at: http://{host}:{port}\n",
            "{details}",
            "   GET  /v1/models           (or /models)\n",
        ),
        title = title,
        version = Engine::version(),
        model_name = model_name,
        host = host,
        port = port,
        details = details,
    );

    if let Err(error) = rocket.launch().await {
        bail!("Server error: {error}");
    }

    Ok(())
}
