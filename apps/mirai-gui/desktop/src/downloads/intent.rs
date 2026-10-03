use std::collections::HashMap;

use tauri::{AppHandle, Manager};
use uzu::{storage::DownloadPhase, types::model::Model};

use super::{DownloadsState, engine, ensure_watcher};

// True = the user left this model downloading, so the next launch resumes it.
// The engine can't tell that apart from a user pause: both restart as Paused.
fn readiness_file() -> crate::error::AppResult<std::path::PathBuf> {
    Ok(crate::storage::mirai_data_dir()?.join("model-readiness.json"))
}

fn read_readiness_file() -> HashMap<String, bool> {
    let Ok(path) = readiness_file() else {
        return HashMap::new();
    };
    let Ok(raw) = std::fs::read_to_string(path) else {
        return HashMap::new();
    };
    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return HashMap::new();
    };
    parsed
        .as_object()
        .map(|map| map.iter().filter_map(|(k, v)| v.as_bool().map(|b| (k.clone(), b))).collect())
        .unwrap_or_default()
}

// The in-memory map is the source of truth for the process; the file is
// written through under the same lock, so concurrent updates cannot lose keys.
static READINESS: std::sync::Mutex<Option<HashMap<String, bool>>> = std::sync::Mutex::new(None);

fn with_readiness<T>(f: impl FnOnce(&mut HashMap<String, bool>) -> T) -> T {
    let mut guard = READINESS.lock().expect("readiness mutex poisoned");
    f(guard.get_or_insert_with(read_readiness_file))
}

fn write_readiness_file(map: &HashMap<String, bool>) -> crate::error::AppResult<()> {
    let json = serde_json::to_string_pretty(map)?;
    crate::storage::write_atomic(&readiness_file()?, json.as_bytes())
}

pub(super) fn readiness_set(
    key: &str,
    value: bool,
) {
    with_readiness(|map| {
        map.insert(key.to_string(), value);
        if let Err(error) = write_readiness_file(map) {
            crate::logger::warn(
                "downloads:readiness-write-failed",
                Some(serde_json::json!({ "key": key, "value": value, "error": error })),
            );
        }
    });
}

pub(crate) fn forget_download_intent(model: &Model) {
    for repo_id in model.repo_ids() {
        readiness_set(&repo_id, false);
    }
}

pub fn auto_resume_on_startup(app: AppHandle) {
    tauri::async_runtime::spawn(async move {
        let Ok(engine) = engine().await else {
            return;
        };
        let wanted: Vec<String> =
            with_readiness(|map| map.iter().filter_map(|(k, v)| v.then_some(k.clone())).collect());
        if wanted.is_empty() {
            return;
        }
        let Ok(models) = engine.models_for_chat().await else {
            return;
        };
        for key in wanted {
            let found = models.iter().filter(|m| m.is_on_device()).find(|m| m.repo_ids().contains(&key));
            let Some(model) = found.cloned() else {
                continue;
            };
            let Some(state) = engine.download_state(&model).await else {
                continue;
            };
            // NotDownloaded with intent set = the cache moved to a new
            // checkpoint_version dir (engine update); re-fetch automatically.
            if !matches!(
                state.phase,
                DownloadPhase::Paused {} | DownloadPhase::Downloading {} | DownloadPhase::NotDownloaded {}
            ) {
                continue;
            }
            let downloads = app.state::<DownloadsState>();
            downloads.remember_one(&model.identifier, &key).await;
            ensure_watcher(app.clone(), &downloads);
            match engine.downloader(&model).resume().await {
                Ok(()) => crate::logger::info(
                    "download:auto-resume",
                    Some(serde_json::json!({ "identifier": model.identifier })),
                ),
                Err(error) => crate::logger::warn(
                    "download:auto-resume-failed",
                    Some(serde_json::json!({ "identifier": model.identifier, "error": error.to_string() })),
                ),
            }
        }
    });
}
