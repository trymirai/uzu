mod events;
mod intent;

use std::{
    collections::HashMap,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};

use events::{DownloadEvent, DownloadStateEvent, announce, events_for};
use futures::StreamExt;
pub use intent::auto_resume_on_startup;
pub(crate) use intent::forget_download_intent;
use intent::readiness_set;
use tauri::{AppHandle, Emitter, Manager};
use uzu::types::model::Model;

use crate::{
    engine::engine,
    error::AppResult,
    models::{PhaseKind, find_downloadable_model},
};

// The event key is the repoId the UI keys download state by.
#[derive(Clone)]
struct RepoInfo {
    event_key: String,
}

#[derive(Default)]
pub struct DownloadsState {
    repo_by_identifier: tokio::sync::Mutex<HashMap<String, RepoInfo>>,
    last_phase: tokio::sync::Mutex<HashMap<String, PhaseKind>>,
    watcher_started: AtomicBool,
    // Orders every emitted event against the snapshot chat_models_get returns,
    // so the client can tell which of the two is newer.
    event_seq: AtomicU64,
}

impl DownloadsState {
    pub(crate) fn event_seq(&self) -> u64 {
        self.event_seq.load(Ordering::SeqCst)
    }

    fn emit(
        &self,
        app: &AppHandle,
        info: &RepoInfo,
        event: DownloadEvent,
    ) {
        let event = DownloadStateEvent {
            seq: self.event_seq.fetch_add(1, Ordering::SeqCst) + 1,
            identifier: info.event_key.clone(),
            event,
        };
        announce(app, &event);
        let _ = app.emit("download-state", event);
    }

    pub async fn remember_repo_ids(
        &self,
        models: &[Model],
    ) {
        let mut map = self.repo_by_identifier.lock().await;
        for model in models {
            let event_key = match model.repo_ids().first() {
                Some(repo_id) => repo_id.clone(),
                None => continue,
            };
            map.insert(
                model.identifier.clone(),
                RepoInfo {
                    event_key,
                },
            );
        }
    }

    async fn repo_info(
        &self,
        identifier: &str,
    ) -> RepoInfo {
        let map = self.repo_by_identifier.lock().await;
        map.get(identifier).cloned().unwrap_or_else(|| RepoInfo {
            event_key: identifier.to_string(),
        })
    }

    async fn remember_one(
        &self,
        identifier: &str,
        event_key: &str,
    ) {
        self.repo_by_identifier.lock().await.insert(
            identifier.to_string(),
            RepoInfo {
                event_key: event_key.to_string(),
            },
        );
    }
}

pub fn ensure_watcher(
    app: AppHandle,
    state: &DownloadsState,
) {
    if state.watcher_started.swap(true, Ordering::SeqCst) {
        return;
    }
    tauri::async_runtime::spawn(async move {
        let engine = match engine().await {
            Ok(engine) => engine,
            Err(error) => {
                crate::logger::error("downloads:watcher:engine", Some(serde_json::json!({ "error": error })));
                app.state::<DownloadsState>().watcher_started.store(false, Ordering::SeqCst);
                return;
            },
        };
        let mut stream = engine.storage_subscribe();
        while let Some(item) = stream.next().await {
            let Ok((identifier, download_state)) = item else {
                continue;
            };
            let downloads = app.state::<DownloadsState>();
            let previous =
                downloads.last_phase.lock().await.insert(identifier.clone(), PhaseKind::from(&download_state.phase));
            let info = downloads.repo_info(&identifier).await;
            for event in events_for(&download_state, previous) {
                downloads.emit(&app, &info, event);
            }
        }
    });
}

async fn run_downloader(
    model: &Model,
    action: impl AsyncFnOnce(uzu::engine::Downloader) -> Result<(), uzu::engine::EngineError>,
) -> AppResult<()> {
    let engine = engine().await?;
    Ok(action(engine.downloader(model)).await?)
}

async fn stop(
    repo_id: &str,
    action: impl AsyncFnOnce(uzu::engine::Downloader) -> Result<(), uzu::engine::EngineError>,
) -> AppResult<()> {
    let model = find_downloadable_model(repo_id).await?;
    run_downloader(&model, action).await?;
    forget_download_intent(&model);
    Ok(())
}

#[tauri::command]
pub async fn download_resume(
    app: AppHandle,
    state: tauri::State<'_, DownloadsState>,
    repo_id: String,
) -> AppResult<()> {
    ensure_watcher(app, &state);
    let model = find_downloadable_model(&repo_id).await?;
    run_downloader(&model, async |d| d.resume().await).await?;
    readiness_set(&repo_id, true);
    Ok(())
}

#[tauri::command]
pub async fn download_pause(repo_id: String) -> AppResult<()> {
    stop(&repo_id, async |d| d.pause().await).await
}

#[tauri::command]
pub async fn download_delete(repo_id: String) -> AppResult<()> {
    stop(&repo_id, async |d| d.delete().await).await
}
