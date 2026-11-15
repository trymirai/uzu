//! Dynamic model loading: the boot-time chat branch of `runner.rs`, extracted
//! and parameterized so the admin API can load models into the slot at runtime.
//!
//! One load = resolve → (download) → thinking support →
//! chat config → VLM tower → chat session → VLM session pool → reply encoding.
//! Decision/SystemOne models are rejected here (boot `--model` keeps the
//! legacy decision branch; see `runner.rs`).

use anyhow::{Context, Result, bail};
use hanashi::chat::hanashi::config::HanashiConfig;
use tokio::sync::Mutex;
use uzu::{
    engine::Engine,
    types::{
        basic::{ContextLength, SamplingSeed},
        model::{Model, ModelSpecialization},
        session::chat::ChatConfig,
    },
};

use crate::{
    common::thinking::ThinkingSupport,
    server::model_slot::LoadedModel,
};

/// Session context length: default cap 128K, CLI-overridable; session creation
/// asserts against the model's RoPE maximum (e.g. 128,000 for VL-2B).
pub const DEFAULT_MAX_CONTEXT_TOKENS: u32 = 131_072;
pub const MAX_CONTEXT_TOKENS_CAP: u32 = 131_072;

/// CLI-derived parameters applied to every model this server loads.
#[derive(Clone, Copy)]
pub struct LoadParams {
    /// Default sampling seed for requests that don't set one.
    pub seed: Option<i64>,
    /// Session context length in tokens, already clamped via
    /// [`clamp_context_tokens`].
    pub max_context_tokens: u32,
}

pub fn clamp_context_tokens(max_context_tokens: Option<u32>) -> u32 {
    max_context_tokens
        .unwrap_or(DEFAULT_MAX_CONTEXT_TOKENS)
        .clamp(1024, MAX_CONTEXT_TOKENS_CAP)
}

/// Resolve and fully load `target` (models-dir directory name, absolute local
/// directory path, or catalog id) into a [`LoadedModel`].
///
/// Must not be called while another load is running or another model is
/// resident-and-busy: swap orchestration (drain + drop + load) is the admin
/// task's job in `model_admin.rs`; this function only builds the model.
pub async fn load_model(engine: &Engine, target: &str, params: &LoadParams) -> Result<LoadedModel> {
    let resolved: Model = crate::common::hf::resolve_model_or_prompt_hf(engine, target, false).await?;

    let local_model_dir = match &resolved.accessibility {
        uzu::types::model::ModelAccessibility::OnDevice {
            source: uzu::types::model::ModelSource::Filesystem { path },
        } => Some(std::path::PathBuf::from(path)),
        _ => None,
    };

    if resolved.specializations.contains(&ModelSpecialization::Decision {}) {
        bail!(
            "Decision/SystemOne models do not support dynamic loading; \
             start the server with `--model {target}` instead"
        );
    }

    tracing::info!("⏳ Preparing model: {}", resolved.identifier);
    prepare_downloads(engine, &resolved).await?;

    let model_name = resolved.identifier.clone();
    let thinking_support = ThinkingSupport::for_model(&resolved);
    let chat_encoding_json = resolved.encoding.as_ref().map(|value| value.json.clone());

    let mut chat_config = params.seed.map_or_else(ChatConfig::default, |seed| {
        ChatConfig::default().with_sampling_seed(SamplingSeed::Custom {
            seed,
        })
    });
    chat_config.context_length = ContextLength::Custom {
        length: params.max_context_tokens as i64,
    };

    let vlm_pipeline = load_vlm_pipeline(&resolved).await;

    let session = engine
        .chat(resolved, chat_config)
        .await
        .context("Failed to create chat session")?;
    let vlm_sessions = std::sync::Arc::new(crate::server::vlm_session::VlmSessionManager::new(
        4,
        std::time::Duration::from_secs(15 * 60),
    ));

    // Resolved before engine.chat consumed the model above: the VLM reply parser
    // runs the same encoding stack the text path resolves for this model.
    // LFM3-VL replies use XML-inline tool calls. The registry may attach the
    // text LFM3 encoding to the bundle, whose LFM2 tool markers do not match,
    // so the VL model always parses with its own config.
    let reply_encoding = if vlm_pipeline.as_ref().is_some_and(|vlm| vlm.is_lfm3_vl()) {
        Some(HanashiConfig::Lfm3Vl)
    } else {
        crate::server::vlm_reply::reply_encoding_from_json(chat_encoding_json.as_deref())
    };
    let encoding_label = match &reply_encoding {
        Some(HanashiConfig::Lfm3Vl) => "lfm3-vl",
        Some(HanashiConfig::Lfm3) => "lfm3",
        Some(HanashiConfig::Qwen35) => "qwen3.5",
        Some(_) => "other",
        None => "none",
    };
    tracing::info!(
        "🧩 VLM reply encoding: {} (bundle encoding.json: {})",
        encoding_label,
        chat_encoding_json.is_some()
    );

    tracing::info!("✅ Model ready: {model_name}");

    // Raw completions need the language model itself (bypassing the chat
    // session); the Metal downcast is guaranteed in this server build.
    let backend_inst = session
        .chat_token_backend_instance()
        .await
        .context("Chat session has no token backend instance")?;
    let uzu_backend = backend_inst
        .as_any()
        .downcast_ref::<uzu::engine::bridge::UzuChatTokenBackendInstance<uzu_engine::backends::metal::Metal>>()
        .context("Chat backend is not Metal")?;
    let lm = std::sync::Arc::clone(uzu_backend.language_model());
    let audio = load_audio_runtime(local_model_dir.as_deref(), &lm).await;

    Ok(LoadedModel {
        model_name,
        session: std::sync::Arc::new(Mutex::new(session)),
        thinking_support,
        vlm_pipeline,
        vlm_sessions,
        audio,
        reply_encoding,
        max_context_tokens: params.max_context_tokens,
        lm,
        loaded_at: chrono::Utc::now(),
        loaded_at_instant: std::time::Instant::now(),
    })
}

/// Download whatever the model needs (empty stream for local directories, so
/// the loop is skipped immediately). Progress goes to tracing instead of the
/// boot spinner — dynamic loads have no TTY and log to the server journal.
pub async fn prepare_downloads(engine: &Engine, resolved: &Model) -> Result<()> {
    let downloader = engine
        .download(resolved)
        .await
        .context("Failed to start model download")?;
    while let Some(update) = downloader.next().await {
        tracing::info!("⏳ Downloading {}: {:.0}%", resolved.identifier, update.progress() * 100.0);
    }
    Ok(())
}

async fn load_audio_runtime(
    model_dir: Option<&std::path::Path>,
    lm: &uzu_engine::engine::language_model::LanguageModel<uzu_engine::backends::metal::Metal>,
) -> Option<std::sync::Arc<Mutex<crate::common::audio::AudioRuntime>>> {
    let dir = model_dir?;
    if !uzu_engine::audio::lfm3_audio::pipeline::Lfm3AudioPipeline::is_lfm3_audio(dir) {
        return None;
    }
    match crate::common::audio::AudioRuntime::load(dir, lm).await {
        Ok(runtime) => {
            tracing::info!("🔊 LFM3-Audio runtime loaded (codec: {})", runtime.codec.is_some());
            Some(std::sync::Arc::new(Mutex::new(runtime)))
        },
        Err(err) => {
            tracing::warn!("Failed to load LFM3-Audio runtime: {err}");
            None
        },
    }
}

/// VLM tower detection and load, copied from the boot branch: a filesystem
/// model with a `vision_config.json` gets a vision pipeline; a failure
/// degrades to text-only with a warning instead of failing the load.
async fn load_vlm_pipeline(
    resolved: &Model,
) -> Option<std::sync::Arc<uzu_engine::vision::AnyVlmPipeline>> {
    match &resolved.accessibility {
        uzu::types::model::ModelAccessibility::OnDevice {
            source: uzu::types::model::ModelSource::Filesystem { path },
        } => {
            let p = std::path::PathBuf::from(path);
            if !p.join("vision_config.json").exists() {
                return None;
            }
            // The VLM prompt renderer needs the tokenizer at load time
            // (special token ids); loading it directly keeps the order
            // independent from the chat session.
            let loaded = tokenizers::Tokenizer::from_file(p.join("tokenizer.json"))
                .map_err(|err| anyhow::anyhow!("Failed to load tokenizer for vision pipeline: {err}"))
                .and_then(|tokenizer| {
                    uzu_engine::vision::AnyVlmPipeline::load_from_model_dir(&p, &tokenizer)
                        .map_err(|err| anyhow::anyhow!("{err}"))
                });
            match loaded {
                Ok(vlm) => {
                    tracing::info!(
                        "📷 Vision tower loaded for multimodal inference (backend: {})",
                        vlm.vision_backend()
                    );
                    Some(std::sync::Arc::new(vlm))
                },
                Err(err) => {
                    tracing::warn!("Failed to load vision tower: {err}");
                    None
                },
            }
        },
        _ => None,
    }
}
