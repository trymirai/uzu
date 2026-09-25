mod gcs;
mod staging;

use std::sync::Arc;

use gcs::{GcsAuth, ensure_auth};
use serde::Serialize;
use staging::{clean_stale_staging, staging_path};
use tauri::{AppHandle, Emitter};
use tauri_plugin_updater::{Update, UpdaterExt};
use tokio::sync::Mutex;

use crate::error::AppResult;

#[derive(Default)]
enum Phase {
    #[default]
    Idle,
    Downloading {
        version: String,
    },
    Downloaded {
        version: String,
    },
    Applying {
        version: String,
    },
}

#[derive(Default)]
pub struct UpdaterState {
    inner: Mutex<UpdaterInner>,
}

#[derive(Default)]
struct UpdaterInner {
    auth: Option<Arc<GcsAuth>>,
    bucket: String,
    phase: Phase,
    pending: Option<Update>,
    // Staging to disk makes a failed apply retryable; the plugin still buffers the
    // whole archive in RAM, so this is not streaming.
    downloaded: Option<(Update, std::path::PathBuf)>,
}

#[derive(Serialize)]
pub struct CheckResult {
    #[serde(rename = "currentVersion")]
    current_version: String,
    #[serde(rename = "latestVersion", skip_serializing_if = "Option::is_none")]
    latest_version: Option<String>,
    #[serde(rename = "hasUpdate")]
    has_update: bool,
    #[serde(rename = "hasDownloadable")]
    has_downloadable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
    source: &'static str,
}

fn current_version(app: &AppHandle) -> String {
    app.package_info().version.to_string()
}

fn check_error(
    current: String,
    reason: String,
) -> CheckResult {
    CheckResult {
        current_version: current,
        latest_version: None,
        has_update: false,
        has_downloadable: false,
        reason: Some(reason),
        source: "error",
    }
}

#[tauri::command]
pub async fn update_check(
    app: AppHandle,
    state: tauri::State<'_, UpdaterState>,
) -> AppResult<CheckResult> {
    let current = current_version(&app);

    {
        let inner = state.inner.lock().await;
        if !matches!(inner.phase, Phase::Idle) {
            return Ok(check_error(current, "update-in-progress".to_string()));
        }
    }

    let (auth, bucket) = match ensure_auth(&state).await {
        Ok(v) => v,
        Err(reason) => return Ok(check_error(current, reason)),
    };

    let endpoint = format!("https://storage.googleapis.com/{bucket}/latest.json");
    let bearer = match auth.bearer().await {
        Ok(b) => b,
        Err(e) => {
            crate::logger::warn("update:error", Some(serde_json::json!({ "phase": "auth", "error": e })));
            return Ok(check_error(current, e));
        },
    };
    let url = match endpoint.parse() {
        Ok(u) => u,
        Err(_) => return Ok(check_error(current, "bad-endpoint".to_string())),
    };
    let updater = match app
        .updater_builder()
        .endpoints(vec![url])
        .and_then(|b| b.header("Authorization", bearer))
        .and_then(|b| b.build())
    {
        Ok(u) => u,
        Err(e) => return Ok(check_error(current, e.to_string())),
    };

    match updater.check().await {
        Ok(Some(update)) => {
            let latest = update.version.clone();
            let mut inner = state.inner.lock().await;
            // A download may have been claimed while check() ran; re-check.
            if !matches!(inner.phase, Phase::Idle) {
                return Ok(check_error(current, "update-in-progress".to_string()));
            }
            inner.pending = Some(update);
            crate::logger::info(
                "update:check",
                Some(serde_json::json!({ "currentVersion": current, "latestVersion": latest, "hasUpdate": true })),
            );
            Ok(CheckResult {
                current_version: current,
                latest_version: Some(latest),
                has_update: true,
                has_downloadable: true,
                reason: None,
                source: "gcs",
            })
        },
        Ok(None) => Ok(CheckResult {
            current_version: current,
            latest_version: None,
            has_update: false,
            has_downloadable: false,
            reason: None,
            source: "gcs",
        }),
        Err(e) => {
            crate::logger::warn("update:error", Some(serde_json::json!({ "phase": "check", "error": e.to_string() })));
            Ok(CheckResult {
                current_version: current,
                latest_version: None,
                has_update: false,
                has_downloadable: false,
                reason: Some(e.to_string()),
                source: "error",
            })
        },
    }
}

#[derive(Serialize)]
pub struct OpResult {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[tauri::command]
pub async fn update_download(
    app: AppHandle,
    state: tauri::State<'_, UpdaterState>,
    version: String,
) -> AppResult<OpResult> {
    // Claim atomically so two calls can't both start a download.
    let update = {
        let mut inner = state.inner.lock().await;
        if !matches!(inner.phase, Phase::Idle) {
            return Ok(OpResult {
                ok: false,
                error: Some("already-in-progress".to_string()),
            });
        }
        let Some(update) = inner.pending.take() else {
            return Ok(OpResult {
                ok: false,
                error: Some("no-pending-update".to_string()),
            });
        };
        if update.version != version {
            inner.pending = Some(update);
            return Ok(OpResult {
                ok: false,
                error: Some("version-mismatch".to_string()),
            });
        }
        inner.phase = Phase::Downloading {
            version: version.clone(),
        };
        update
    };

    // The staging path lives only in process state, so a crashed run orphans its
    // file; sweep before writing a new one.
    clean_stale_staging();

    let staged = match update.download(|_, _| {}, || {}).await {
        Ok(bytes) => {
            let path = staging_path(&version);
            std::fs::write(&path, &bytes).map(|_| path).map_err(|e| e.to_string())
        },
        Err(e) => Err(e.to_string()),
    };
    match staged {
        Ok(path) => {
            let mut inner = state.inner.lock().await;
            inner.phase = Phase::Downloaded {
                version: version.clone(),
            };
            inner.downloaded = Some((update, path));
            drop(inner);
            crate::logger::info("update:download:done", Some(serde_json::json!({ "version": version })));
            let _ = app.emit("update-download-done", serde_json::json!({ "version": version }));
            Ok(OpResult {
                ok: true,
                error: None,
            })
        },
        Err(e) => {
            let mut inner = state.inner.lock().await;
            inner.phase = Phase::Idle;
            inner.pending = Some(update);
            drop(inner);
            crate::logger::warn(
                "update:error",
                Some(serde_json::json!({ "phase": "download", "version": version, "error": e.to_string() })),
            );
            let _ =
                app.emit("update-download-error", serde_json::json!({ "version": version, "error": e.to_string() }));
            Ok(OpResult {
                ok: false,
                error: Some(e.to_string()),
            })
        },
    }
}

#[tauri::command]
pub async fn update_apply(
    app: AppHandle,
    state: tauri::State<'_, UpdaterState>,
) -> AppResult<OpResult> {
    // Claim under one lock so a double-click can't run two installs that both
    // rename the .app; restore on failure so apply stays retryable.
    let (update, path, version) = {
        let mut inner = state.inner.lock().await;
        let version = match &inner.phase {
            Phase::Downloaded {
                version,
            } => version.clone(),
            Phase::Applying {
                ..
            } => {
                return Ok(OpResult {
                    ok: false,
                    error: Some("already-applying".to_string()),
                });
            },
            _ => {
                return Ok(OpResult {
                    ok: false,
                    error: Some("no-downloaded-update".to_string()),
                });
            },
        };
        let Some((update, path)) = inner.downloaded.take() else {
            return Ok(OpResult {
                ok: false,
                error: Some("no-downloaded-update".to_string()),
            });
        };
        inner.phase = Phase::Applying {
            version: version.clone(),
        };
        (update, path, version)
    };

    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            restore_downloaded(&state, update, path, version).await;
            return Ok(OpResult {
                ok: false,
                error: Some(e.to_string()),
            });
        },
    };
    match update.install(bytes) {
        Ok(()) => {
            let _ = std::fs::remove_file(&path);
            crate::logger::info("update:apply", None);
            app.restart();
        },
        Err(e) => {
            crate::logger::error("update:apply:error", Some(serde_json::json!({ "error": e.to_string() })));
            restore_downloaded(&state, update, path, version).await;
            Ok(OpResult {
                ok: false,
                error: Some(e.to_string()),
            })
        },
    }
}

async fn restore_downloaded(
    state: &UpdaterState,
    update: Update,
    path: std::path::PathBuf,
    version: String,
) {
    let mut inner = state.inner.lock().await;
    inner.phase = Phase::Downloaded {
        version,
    };
    inner.downloaded = Some((update, path));
}

#[derive(Serialize)]
#[serde(tag = "phase", rename_all = "lowercase")]
pub enum StatusResult {
    Idle,
    Downloading {
        version: String,
    },
    Downloaded {
        version: String,
    },
}

#[tauri::command]
pub async fn update_status(state: tauri::State<'_, UpdaterState>) -> AppResult<StatusResult> {
    let inner = state.inner.lock().await;
    Ok(match &inner.phase {
        Phase::Idle => StatusResult::Idle,
        Phase::Downloading {
            version,
        } => StatusResult::Downloading {
            version: version.clone(),
        },
        Phase::Downloaded {
            version,
        }
        | Phase::Applying {
            version,
        } => StatusResult::Downloaded {
            version: version.clone(),
        },
    })
}
