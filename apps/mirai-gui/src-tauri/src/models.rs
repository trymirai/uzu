use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

use hanashi::chat::EncodingConfig;
use serde::Serialize;
use uzu::{
    storage::{DownloadPhase, DownloadState},
    types::{
        basic::{Image, ImageFormat, ImageTheme, ReasoningEffort},
        model::{Model, ModelVendor},
    },
};

use crate::{
    engine::engine,
    error::{AppError, AppResult},
};

#[derive(Serialize, Clone, Debug, PartialEq, Default)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum ReasoningSupport {
    #[default]
    Unsupported,
    AlwaysOn,
    Toggle,
    Levels {
        efforts: Vec<ReasoningEffort>,
    },
}

impl ReasoningSupport {
    pub fn effective(
        &self,
        requested: Option<ReasoningEffort>,
    ) -> Option<ReasoningEffort> {
        match self {
            ReasoningSupport::Unsupported | ReasoningSupport::AlwaysOn => None,
            ReasoningSupport::Toggle => Some(if requested == Some(ReasoningEffort::Disabled) {
                ReasoningEffort::Disabled
            } else {
                ReasoningEffort::Default
            }),
            ReasoningSupport::Levels {
                efforts,
            } => requested
                .filter(|effort| efforts.contains(effort))
                .or_else(|| efforts.contains(&ReasoningEffort::Default).then_some(ReasoningEffort::Default))
                // hanashi rejects an effort missing from the mapping, so never invent one.
                .or_else(|| efforts.iter().copied().find(|effort| *effort != ReasoningEffort::Disabled)),
        }
    }

    pub fn cheapest(&self) -> Option<ReasoningEffort> {
        match self {
            ReasoningSupport::Unsupported | ReasoningSupport::AlwaysOn => None,
            ReasoningSupport::Toggle => Some(ReasoningEffort::Disabled),
            ReasoningSupport::Levels {
                efforts,
            } => efforts.iter().copied().find(|e| *e != ReasoningEffort::Default),
        }
    }
}

// uzu lists efforts in template-mapping order, which differs per model.
const EFFORT_ORDER: [ReasoningEffort; 6] = [
    ReasoningEffort::Disabled,
    ReasoningEffort::Default,
    ReasoningEffort::Low,
    ReasoningEffort::Medium,
    ReasoningEffort::High,
    ReasoningEffort::XHigh,
];

// Resolving an encoding parses four bundled configs; models share a handful of variants.
static ENCODING_SUPPORT: OnceLock<Mutex<HashMap<String, Option<ReasoningSupport>>>> = OnceLock::new();

fn reasoning_support_from_encoding(model: &Model) -> Option<ReasoningSupport> {
    let encoding = model.encoding.as_ref()?;
    let cache = ENCODING_SUPPORT.get_or_init(Default::default);
    if let Some(cached) = cache.lock().expect("encoding cache poisoned").get(&encoding.json) {
        return cached.clone();
    }
    let support = parse_reasoning_support(model, &encoding.json);
    cache.lock().expect("encoding cache poisoned").insert(encoding.json.clone(), support.clone());
    support
}

fn parse_reasoning_support(
    model: &Model,
    encoding_json: &str,
) -> Option<ReasoningSupport> {
    let config = match serde_json::from_str::<EncodingConfig>(encoding_json) {
        Ok(config) => config,
        Err(error) => {
            crate::logger::warn(
                "models:encoding:parse-error",
                Some(serde_json::json!({ "identifier": model.identifier, "error": error.to_string() })),
            );
            return None;
        },
    };
    let capabilities = match config.capabilities() {
        Ok(capabilities) => capabilities,
        Err(error) => {
            crate::logger::warn(
                "models:encoding:capabilities-error",
                Some(serde_json::json!({ "identifier": model.identifier, "error": error.to_string() })),
            );
            return None;
        },
    };
    if !capabilities.supports_reasoning {
        return Some(ReasoningSupport::Unsupported);
    }
    let has_levels = capabilities.reasoning_efforts.iter().any(|effort| {
        matches!(
            effort,
            ReasoningEffort::Low | ReasoningEffort::Medium | ReasoningEffort::High | ReasoningEffort::XHigh
        )
    });
    if has_levels {
        let efforts =
            EFFORT_ORDER.into_iter().filter(|effort| capabilities.reasoning_efforts.contains(effort)).collect();
        Some(ReasoningSupport::Levels {
            efforts,
        })
    } else if capabilities.supports_disable_reasoning {
        Some(ReasoningSupport::Toggle)
    } else {
        Some(ReasoningSupport::AlwaysOn)
    }
}

pub fn reasoning_support(model: &Model) -> ReasoningSupport {
    reasoning_support_from_encoding(model).unwrap_or_default()
}

// Serialized by variant name; the client matches on these strings.
#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhaseKind {
    NotDownloaded,
    Downloading,
    Paused,
    Downloaded,
    Locked,
    Error,
}

impl From<&DownloadPhase> for PhaseKind {
    fn from(phase: &DownloadPhase) -> Self {
        match phase {
            DownloadPhase::NotDownloaded {} => PhaseKind::NotDownloaded,
            DownloadPhase::Downloading {} => PhaseKind::Downloading,
            DownloadPhase::Paused {} => PhaseKind::Paused,
            DownloadPhase::Downloaded {} => PhaseKind::Downloaded,
            DownloadPhase::Locked {
                ..
            } => PhaseKind::Locked,
            DownloadPhase::Error {
                ..
            } => PhaseKind::Error,
        }
    }
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ModelDownloadState {
    pub total_kbytes: u64,
    pub downloaded_kbytes: u64,
    pub phase: PhaseKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    // Event sequence at the time of the snapshot; download events carry theirs.
    pub seq: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineModel {
    pub identifier: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repo_id: Option<String>,
    pub vendor: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vendor_icons: Option<VendorIcons>,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family_identifier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param_size: Option<i64>,
    pub reasoning: ReasoningSupport,
    pub quantization: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization_bits: Option<u32>,
    pub state: ModelDownloadState,
}

// Models served from a plain directory (LOCAL_PATH) have nothing to download.
pub fn model_download_state(
    model: &Model,
    state: Option<&DownloadState>,
    seq: u64,
) -> ModelDownloadState {
    if model.is_downloadable() {
        map_download_state(state, seq)
    } else {
        ModelDownloadState {
            total_kbytes: 0,
            downloaded_kbytes: 0,
            phase: PhaseKind::Downloaded,
            error: None,
            seq,
        }
    }
}

pub fn map_download_state(
    state: Option<&DownloadState>,
    seq: u64,
) -> ModelDownloadState {
    let Some(state) = state else {
        return ModelDownloadState {
            total_kbytes: 0,
            downloaded_kbytes: 0,
            phase: PhaseKind::NotDownloaded,
            error: None,
            seq,
        };
    };
    let error = match &state.phase {
        DownloadPhase::Error {
            message,
        } => Some(message.clone()),
        _ => None,
    };
    ModelDownloadState {
        total_kbytes: (state.total_bytes.max(0) as u64) / 1024,
        downloaded_kbytes: (state.downloaded_bytes.max(0) as u64) / 1024,
        phase: PhaseKind::from(&state.phase),
        error,
        seq,
    }
}

#[derive(Serialize)]
pub struct VendorIcons {
    pub light: String,
    pub dark: String,
}

// The registry serves each icon as SVG and PNG per theme; SVG scales cleanly.
fn icon_url(
    icons: &[Image],
    theme: ImageTheme,
) -> Option<String> {
    let of_theme = |format: ImageFormat| {
        icons.iter().find(|icon| icon.theme == theme && icon.format == format).map(|icon| icon.url.clone())
    };
    of_theme(ImageFormat::Svg).or_else(|| of_theme(ImageFormat::Png)).or_else(|| of_theme(ImageFormat::Webp))
}

fn vendor_icons(vendor: &ModelVendor) -> Option<VendorIcons> {
    let icons = &vendor.metadata.icons;
    let light = icon_url(icons, ImageTheme::Light);
    let dark = icon_url(icons, ImageTheme::Dark);
    Some(VendorIcons {
        light: light.clone().or_else(|| dark.clone())?,
        dark: dark.or(light)?,
    })
}

pub fn engine_model(
    model: &Model,
    state: ModelDownloadState,
    reasoning: ReasoningSupport,
) -> EngineModel {
    let vendor = model.family.as_ref().map(|f| &f.vendor).or_else(|| model.quantization.as_ref().map(|q| &q.vendor));
    EngineModel {
        identifier: model.identifier.clone(),
        repo_id: model.repo_ids().first().cloned(),
        vendor: vendor.map(|v| v.name()).unwrap_or_default(),
        vendor_icons: vendor.and_then(vendor_icons),
        name: model.name(),
        family_identifier: model.family.as_ref().map(|f| f.identifier.clone()),
        family_name: model.family.as_ref().map(|f| f.name()),
        param_size: model.properties.as_ref().map(|p| p.size),
        reasoning,
        quantization: model.quantization.as_ref().map(|q| q.method.clone()),
        quantization_bits: model.quantization.as_ref().map(|q| q.bits_per_weight),
        state,
    }
}

pub fn find_in(
    models: Vec<Model>,
    repo_id: &str,
) -> Option<Model> {
    let local: Vec<Model> = models.into_iter().filter(|m| m.is_local()).collect();
    local
        .iter()
        .find(|m| m.repo_ids().first().is_some_and(|id| id == repo_id))
        .or_else(|| local.iter().find(|m| m.repo_ids().iter().any(|id| id == repo_id)))
        // LOCAL_PATH models have no repo id; the client keys them by identifier.
        .or_else(|| local.iter().find(|m| m.identifier == repo_id))
        .cloned()
}

// List-based lookup on purpose: engine.model() can mis-resolve when provider
// keys are registered.
pub async fn find_chat_model(repo_id: &str) -> AppResult<Model> {
    let engine = engine().await?;
    let models = engine.models_for_chat().await?;
    find_in(models, repo_id).ok_or_else(|| AppError::msg(format!("Model not found: {repo_id}")))
}

// Cleanup enumerates by engine identifier (download_states' key), not repo_id.
// Searches every engine list, not just chat: the model cache is shared with
// other uzu-based apps, so it can hold weights this app never downloads.
pub async fn find_model_by_identifier(identifier: &str) -> Option<Model> {
    let engine = engine().await.ok()?;
    for models in [
        engine.models_for_chat().await.ok(),
        engine.models_for_text_to_speech().await.ok(),
        engine.models_for_classification().await.ok(),
    ]
    .into_iter()
    .flatten()
    {
        if let Some(model) = models.into_iter().find(|m| m.identifier == identifier) {
            return Some(model);
        }
    }
    None
}

pub async fn find_downloadable_model(key: &str) -> AppResult<Model> {
    let engine = engine().await?;
    let chat = engine.models_for_chat().await?;
    if let Some(model) = find_in(chat, key) {
        return Ok(model);
    }
    find_model_by_identifier(key).await.ok_or_else(|| AppError::msg(format!("Model not found: {key}")))
}

#[tauri::command]
pub async fn chat_models_get(
    app: tauri::AppHandle,
    state: tauri::State<'_, crate::downloads::DownloadsState>,
) -> AppResult<Vec<EngineModel>> {
    let engine = engine().await?;
    let models = engine.models_for_chat().await?;
    // Taken before the states: an event emitted while they are read then
    // carries a newer seq than this snapshot and wins on the client.
    let seq = state.event_seq();
    let states = engine.download_states().await;
    let local: Vec<Model> = models.into_iter().filter(|m| m.is_local()).collect();
    state.remember_repo_ids(&local).await;
    crate::downloads::ensure_watcher(app, &state);
    let mut result = Vec::with_capacity(local.len());
    for model in &local {
        let reasoning = reasoning_support(model);
        let state = model_download_state(model, states.get(&model.identifier), seq);
        result.push(engine_model(model, state, reasoning));
    }
    Ok(result)
}
