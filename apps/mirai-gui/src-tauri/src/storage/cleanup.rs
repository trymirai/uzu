use std::path::{Path, PathBuf};

use super::chats_dir;
use crate::error::AppResult;

#[derive(serde::Serialize, Default)]
#[serde(rename_all = "camelCase")]
struct CategorySize {
    count: u64,
    size_bytes: u64,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CleanupPreview {
    dialogs: CategorySize,
    models: ModelsCategory,
    logs: LogsCategory,
}

#[derive(serde::Serialize, Default)]
#[serde(rename_all = "camelCase")]
struct ModelsCategory {
    count: u64,
    size_bytes: u64,
}

#[derive(serde::Serialize, Default)]
#[serde(rename_all = "camelCase")]
struct LogsCategory {
    size_bytes: u64,
}

fn log_files() -> Vec<PathBuf> {
    [crate::logger::log_file_path(), crate::logger::rotated_log_file_path()].into_iter().flatten().collect()
}

fn logs_size() -> LogsCategory {
    let size_bytes =
        log_files().iter().filter_map(|p| std::fs::metadata(p).ok()).filter(|m| m.is_file()).map(|m| m.len()).sum();
    LogsCategory {
        size_bytes,
    }
}

fn dir_size(
    dir: &Path,
    ext: Option<&str>,
) -> CategorySize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return CategorySize::default();
    };
    let mut out = CategorySize::default();
    for entry in entries.flatten() {
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        if let Some(ext) = ext
            && !entry.file_name().to_str().is_some_and(|n| n.ends_with(ext))
        {
            continue;
        }
        out.count += 1;
        out.size_bytes += meta.len();
    }
    out
}

// The UI passes resident-session repoIds as the skip list, while
// download_states is keyed by engine identifier; match against both.
async fn is_skipped(
    identifier: &str,
    skip: &[String],
) -> bool {
    if skip.is_empty() {
        return false;
    }
    if skip.iter().any(|s| s == identifier) {
        return true;
    }
    match crate::models::find_model_by_identifier(identifier).await {
        Some(model) => model.repo_ids().iter().any(|repo_id| skip.iter().any(|s| s == repo_id)),
        None => false,
    }
}

async fn downloaded_models(skip: &[String]) -> ModelsCategory {
    let mut models = ModelsCategory::default();
    let Ok(engine) = crate::engine::engine().await else {
        return models;
    };
    let states = engine.download_states().await;
    for (identifier, state) in states.iter() {
        if !matches!(state.phase, uzu::storage::DownloadPhase::Downloaded {}) || is_skipped(identifier, skip).await {
            continue;
        }
        models.count += 1;
        models.size_bytes += state.total_bytes.max(0) as u64;
    }
    models
}

#[tauri::command]
pub async fn cleanup_preview(skip_model_identifiers: Vec<String>) -> AppResult<CleanupPreview> {
    Ok(CleanupPreview {
        dialogs: dir_size(&chats_dir()?, Some(".md")),
        models: downloaded_models(&skip_model_identifiers).await,
        logs: logs_size(),
    })
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum Category {
    Dialogs,
    Models,
    Logs,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CleanupResult {
    executed: Vec<Category>,
    models_skipped: Vec<String>,
}

fn clear_dir(
    dir: &Path,
    ext: Option<&str>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if entry.metadata().map(|m| m.is_file()).unwrap_or(false) {
            if let Some(ext) = ext
                && !entry.file_name().to_str().is_some_and(|n| n.ends_with(ext))
            {
                continue;
            }
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

#[tauri::command]
pub async fn cleanup_execute(
    categories: Vec<Category>,
    skip_model_identifiers: Vec<String>,
) -> AppResult<CleanupResult> {
    let skip = skip_model_identifiers;
    let mut executed = Vec::new();
    let mut models_skipped = Vec::new();

    if categories.contains(&Category::Dialogs) {
        clear_dir(&chats_dir()?, Some(".md"));
        executed.push(Category::Dialogs);
    }

    if categories.contains(&Category::Models) {
        let mut all_deleted = false;
        if let Ok(engine) = crate::engine::engine().await {
            all_deleted = true;
            let states = engine.download_states().await;
            for (identifier, state) in states.iter() {
                let is_active = matches!(
                    state.phase,
                    uzu::storage::DownloadPhase::Downloading {}
                        | uzu::storage::DownloadPhase::Paused {}
                        | uzu::storage::DownloadPhase::Locked { .. }
                );
                let is_removable = matches!(
                    state.phase,
                    uzu::storage::DownloadPhase::Downloaded {} | uzu::storage::DownloadPhase::Error { .. }
                );
                if is_active || is_skipped(identifier, &skip).await {
                    models_skipped.push(identifier.clone());
                } else if is_removable && let Some(model) = crate::models::find_model_by_identifier(identifier).await {
                    match engine.downloader(&model).delete().await {
                        Ok(()) => crate::downloads::forget_download_intent(&model),
                        Err(error) => {
                            crate::logger::error(
                                "cleanup:model-delete",
                                Some(serde_json::json!({ "identifier": identifier, "error": error.to_string() })),
                            );
                            all_deleted = false;
                        },
                    }
                }
            }
        }
        if all_deleted {
            executed.push(Category::Models);
        }
    }

    if categories.contains(&Category::Logs) {
        for path in log_files() {
            let _ = std::fs::remove_file(path);
        }
        executed.push(Category::Logs);
    }

    Ok(CleanupResult {
        executed,
        models_skipped,
    })
}
