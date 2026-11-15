use std::path::Path;

use rocket::{State, get, serde::json::Json};
use serde::{Deserialize, Serialize};

use crate::server::{ServerState, model_slot::SlotSnapshot, request_info::RequestInfo};

#[derive(Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    /// Uzu extension: whether this model is resident in the server's slot.
    pub loaded: bool,
}

#[derive(Serialize, Deserialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<Model>,
}

/// Subdirectories of `dir` that look like models (contain a `config.json`).
fn scan_models_dir(dir: Option<&Path>) -> Vec<String> {
    let Some(dir) = dir else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        if dir.exists() {
            tracing::warn!("--models-dir {} is unreadable", dir.display());
        }
        return Vec::new();
    };
    let mut ids: Vec<String> = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_dir() && path.join("config.json").is_file())
        .filter_map(|path| path.file_name().map(|name| name.to_string_lossy().into_owned()))
        .collect();
    ids.sort();
    ids
}

/// OpenAI-compatible model list: the models-dir scan + local registry models
/// (HF cache, ~/.cache/mirai/models, LOCAL_PATH) + the resident model.
#[get("/models")]
#[tracing::instrument(skip_all, parent = &request_info.span)]
pub async fn handle_models(
    chat: &State<crate::server::ChatState>,
    request_info: &RequestInfo,
) -> Json<ModelsResponse> {
    let mut ids = scan_models_dir(chat.models_dir.as_deref());
    if let Ok(models) = chat.engine.models().await {
        for model in models {
            if matches!(
                model.accessibility,
                uzu::types::model::ModelAccessibility::OnDevice {
                    source: uzu::types::model::ModelSource::Filesystem { .. },
                }
            ) && !ids.contains(&model.identifier)
            {
                ids.push(model.identifier);
            }
        }
        ids.sort();
    }
    let mut resident = Option::<String>::None;
    if let SlotSnapshot::Ready(loaded) = chat.slot.snapshot().await {
        resident = Some(loaded.model_name.clone());
        if !ids.contains(&loaded.model_name) {
            ids.push(loaded.model_name.clone());
            ids.sort();
        }
    }
    Json(ModelsResponse {
        object: "list".to_string(),
        data: ids
            .into_iter()
            .map(|id| Model {
                loaded: resident.as_deref() == Some(id.as_str()),
                id,
                object: "model".to_string(),
                created: 0,
                owned_by: "uzu".to_string(),
            })
            .collect(),
    })
}

/// Single-model list for the legacy `--model` boot branches (decision
/// servers never manage a slot). Kept wire-compatible with `handle_models`.
#[get("/models")]
#[tracing::instrument(skip_all, parent = &request_info.span)]
pub fn handle_models_static(state: &State<ServerState>, request_info: &RequestInfo) -> Json<ModelsResponse> {
    let data = state
        .model_name
        .iter()
        .map(|name| Model {
            id: name.clone(),
            object: "model".to_string(),
            created: 0,
            owned_by: "uzu".to_string(),
            loaded: true,
        })
        .collect();
    Json(ModelsResponse {
        object: "list".to_string(),
        data,
    })
}
