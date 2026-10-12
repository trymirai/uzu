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
        session::chat::ChatModelCapabilities,
    },
};

use crate::{
    engine::engine,
    error::{AppError, AppResult},
};

#[derive(Serialize, Clone, Debug, PartialEq, Default)]
#[serde(tag = "kind", rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum ReasoningSupport {
    #[default]
    Unsupported,
    AlwaysOn,
    Toggle {
        default_effort: ReasoningEffort,
    },
    Levels {
        efforts: Vec<ReasoningEffort>,
        #[serde(skip_serializing_if = "Option::is_none")]
        default_effort: Option<ReasoningEffort>,
    },
}

impl ReasoningSupport {
    pub fn effective(
        &self,
        requested: Option<ReasoningEffort>,
    ) -> Option<ReasoningEffort> {
        match self {
            ReasoningSupport::Unsupported | ReasoningSupport::AlwaysOn => None,
            ReasoningSupport::Toggle {
                ..
            } => requested.filter(|effort| *effort == ReasoningEffort::Disabled),
            ReasoningSupport::Levels {
                efforts,
                ..
            } => requested
                // Leave default selection to the template, including mappings
                // whose default is not named as an explicit effort level.
                .filter(|effort| *effort != ReasoningEffort::Default && efforts.contains(effort)),
        }
    }

    pub fn cheapest(&self) -> Option<ReasoningEffort> {
        match self {
            ReasoningSupport::Unsupported | ReasoningSupport::AlwaysOn => None,
            ReasoningSupport::Toggle {
                ..
            } => Some(ReasoningEffort::Disabled),
            ReasoningSupport::Levels {
                efforts,
                ..
            } => efforts.iter().copied().find(|e| *e != ReasoningEffort::Default),
        }
    }
}

// uzu lists efforts in template-mapping order, which differs per model.
const EFFORT_ORDER: [ReasoningEffort; 5] = [
    ReasoningEffort::Disabled,
    ReasoningEffort::Low,
    ReasoningEffort::Medium,
    ReasoningEffort::High,
    ReasoningEffort::XHigh,
];

// Resolving an encoding parses four bundled configs; models share a handful of variants.
#[derive(Clone)]
struct EncodingSupport {
    reasoning: ReasoningSupport,
    tools: bool,
}

static ENCODING_SUPPORT: OnceLock<Mutex<HashMap<String, Option<EncodingSupport>>>> = OnceLock::new();

fn encoding_support(model: &Model) -> Option<EncodingSupport> {
    let encoding = model.encoding.as_ref()?;
    let cache = ENCODING_SUPPORT.get_or_init(Default::default);
    if let Some(cached) = cache.lock().expect("encoding cache poisoned").get(&encoding.json) {
        return cached.clone();
    }
    let support = parse_encoding_support(model, &encoding.json);
    cache.lock().expect("encoding cache poisoned").insert(encoding.json.clone(), support.clone());
    support
}

fn parse_encoding_support(
    model: &Model,
    encoding_json: &str,
) -> Option<EncodingSupport> {
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
    Some(EncodingSupport {
        reasoning: classify_reasoning_support(&capabilities, config.default_reasoning_effort().ok().flatten()),
        tools: capabilities.supports_tools,
    })
}

fn classify_reasoning_support(
    capabilities: &ChatModelCapabilities,
    default_effort: Option<ReasoningEffort>,
) -> ReasoningSupport {
    let has_levels = capabilities.reasoning_efforts.iter().any(|effort| {
        matches!(
            effort,
            ReasoningEffort::Low | ReasoningEffort::Medium | ReasoningEffort::High | ReasoningEffort::XHigh
        )
    });
    if !capabilities.supports_reasoning {
        ReasoningSupport::Unsupported
    } else if has_levels {
        let efforts: Vec<_> =
            EFFORT_ORDER.into_iter().filter(|effort| capabilities.reasoning_efforts.contains(effort)).collect();
        ReasoningSupport::Levels {
            default_effort: default_effort.filter(|effort| efforts.contains(effort)),
            efforts,
        }
    } else if capabilities.supports_disable_reasoning {
        if default_effort != Some(ReasoningEffort::Default)
            || !capabilities.reasoning_efforts.contains(&ReasoningEffort::Default)
        {
            // Identical or unresolved modes do not prove an enabled choice.
            ReasoningSupport::Levels {
                efforts: vec![ReasoningEffort::Disabled],
                default_effort: default_effort.filter(|effort| *effort == ReasoningEffort::Disabled),
            }
        } else {
            ReasoningSupport::Toggle {
                default_effort: ReasoningEffort::Default,
            }
        }
    } else {
        ReasoningSupport::AlwaysOn
    }
}

pub fn reasoning_support(model: &Model) -> ReasoningSupport {
    encoding_support(model).map(|support| support.reasoning).unwrap_or_default()
}

// Serialized by variant name; the client matches on these strings.
#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhaseKind {
    Initializing,
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
            DownloadPhase::Initializing {} => PhaseKind::Initializing,
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
    pub supports_tools: bool,
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
            phase: PhaseKind::Initializing,
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
        supports_tools: encoding_support(model).is_some_and(|support| support.tools),
        quantization: model.quantization.as_ref().map(|q| q.method.clone()),
        quantization_bits: model.quantization.as_ref().map(|q| q.bits_per_weight),
        state,
    }
}

pub fn find_in(
    models: Vec<Model>,
    repo_id: &str,
) -> Option<Model> {
    let local: Vec<Model> = models.into_iter().filter(|m| m.is_on_device()).collect();
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
) -> AppResult<ModelCatalog> {
    let engine = engine().await?;
    crate::downloads::ensure_watcher(app, &state);
    let (models, complete) = engine.catalog_snapshot().await?;
    // Taken before the states: an event emitted while they are read then
    // carries a newer seq than this snapshot and wins on the client.
    let seq = state.event_seq();
    let states = engine.download_states().await;
    let local: Vec<Model> = models.into_iter().filter(|m| m.is_on_device() && m.is_chat_capable()).collect();
    state.remember_repo_ids(&local).await;
    let mut result = Vec::with_capacity(local.len());
    for model in &local {
        let reasoning = reasoning_support(model);
        let state = model_download_state(model, states.get(&model.identifier), seq);
        result.push(engine_model(model, state, reasoning));
    }
    Ok(ModelCatalog {
        models: result,
        complete,
        refreshing: engine.catalog_is_refreshing(),
    })
}

#[tauri::command]
pub async fn chat_models_refresh() -> AppResult<()> {
    engine().await?.refresh_catalog().await?;
    Ok(())
}

#[derive(Serialize)]
pub struct ModelCatalog {
    pub models: Vec<EngineModel>,
    pub complete: bool,
    pub refreshing: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn support(json: &str) -> ReasoningSupport {
        let config: EncodingConfig = serde_json::from_str(json).unwrap();
        classify_reasoning_support(&config.capabilities().unwrap(), config.default_reasoning_effort().unwrap())
    }

    #[test]
    fn model_level_metadata_exposes_only_real_choices_and_actual_default() {
        assert_eq!(
            serde_json::to_value(support(r#"{"type":"hanashi","name":"qwen3.8"}"#)).unwrap(),
            serde_json::json!({"kind":"levels", "efforts":["disabled","low","medium","xhigh"], "defaultEffort":"xhigh"})
        );
        assert_eq!(
            serde_json::to_value(support(r#"{"type":"hanashi","name":"muse-glimmer"}"#)).unwrap(),
            serde_json::json!({"kind":"levels", "efforts":["low","medium","high","xhigh"], "defaultEffort":"high"})
        );
        for encoding in ["hanashi", "harmony"] {
            assert_eq!(
                serde_json::to_value(support(&format!(r#"{{"type":"{encoding}","name":"gpt-oss"}}"#))).unwrap(),
                serde_json::json!({"kind":"levels", "efforts":["low","medium","high"], "defaultEffort":"medium"})
            );
        }
    }

    #[test]
    fn model_toggle_metadata_uses_the_enabled_default() {
        for name in ["qwen3", "qwen3.5", "qwen3.6", "gemma-4"] {
            assert_eq!(
                serde_json::to_value(support(&format!(r#"{{"type":"hanashi","name":"{name}"}}"#))).unwrap(),
                serde_json::json!({"kind":"toggle", "defaultEffort":"default"})
            );
        }
    }

    #[test]
    fn identical_toggle_modes_do_not_advertise_a_switch() {
        let config: EncodingConfig = serde_json::from_str(r#"{"type":"hanashi","name":"qwen3.5"}"#).unwrap();
        for default in [None, Some(ReasoningEffort::Disabled)] {
            assert_eq!(
                classify_reasoning_support(&config.capabilities().unwrap(), default),
                ReasoningSupport::Levels {
                    efforts: vec![ReasoningEffort::Disabled],
                    default_effort: default
                }
            );
        }
        let mut capabilities = config.capabilities().unwrap();
        capabilities.reasoning_efforts.retain(|effort| *effort != ReasoningEffort::Default);
        assert_eq!(
            classify_reasoning_support(&capabilities, Some(ReasoningEffort::Default)),
            ReasoningSupport::Levels {
                efforts: vec![ReasoningEffort::Disabled],
                default_effort: None
            }
        );
    }

    #[test]
    fn level_mapping_without_default_keeps_the_template_default_unset() {
        let support = ReasoningSupport::Levels {
            efforts: vec![ReasoningEffort::Low, ReasoningEffort::Medium, ReasoningEffort::High],
            default_effort: None,
        };
        assert_eq!(support.effective(None), None);
        assert_eq!(support.effective(Some(ReasoningEffort::Default)), None);
        assert_eq!(support.effective(Some(ReasoningEffort::Medium)), Some(ReasoningEffort::Medium));
        assert_eq!(support.effective(Some(ReasoningEffort::Disabled)), None);
        assert_eq!(support.effective(Some(ReasoningEffort::XHigh)), None);
    }

    #[test]
    fn explicit_levels_are_retained_but_default_is_not_an_override() {
        let support = ReasoningSupport::Levels {
            efforts: vec![ReasoningEffort::Disabled, ReasoningEffort::XHigh],
            default_effort: Some(ReasoningEffort::XHigh),
        };
        assert_eq!(support.effective(None), None);
        assert_eq!(support.effective(Some(ReasoningEffort::Default)), None);
        assert_eq!(support.effective(Some(ReasoningEffort::Disabled)), Some(ReasoningEffort::Disabled));
        assert_eq!(support.effective(Some(ReasoningEffort::XHigh)), Some(ReasoningEffort::XHigh));
    }

    #[test]
    fn toggles_and_fixed_reasoning_modes_only_apply_supported_overrides() {
        let toggle = ReasoningSupport::Toggle {
            default_effort: ReasoningEffort::Default,
        };
        assert_eq!(toggle.effective(None), None);
        assert_eq!(toggle.effective(Some(ReasoningEffort::Default)), None);
        assert_eq!(toggle.effective(Some(ReasoningEffort::High)), None);
        assert_eq!(toggle.effective(Some(ReasoningEffort::Disabled)), Some(ReasoningEffort::Disabled));
        for fixed in [ReasoningSupport::Unsupported, ReasoningSupport::AlwaysOn] {
            assert_eq!(fixed.effective(None), None);
            assert_eq!(fixed.effective(Some(ReasoningEffort::Default)), None);
            assert_eq!(fixed.effective(Some(ReasoningEffort::Disabled)), None);
        }
    }
}
