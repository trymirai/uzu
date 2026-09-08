use std::{
    collections::HashMap,
    error::Error,
    path::PathBuf,
    sync::{Arc, LazyLock, Mutex, PoisonError},
};

use download_manager::{
    DownloadError, DownloadManager, DownloadManagerType, DownloadPhase, DownloadTask, DownloadTaskRequest,
};
use kiban::{eprintf, printf, rt::RuntimeHandle};
use tokio::sync::OnceCell;
use tokio_stream::StreamExt;
use wasm_bindgen::{JsError, JsValue, prelude::wasm_bindgen};

static MANAGER: OnceCell<DownloadManager> = OnceCell::const_new();
static TASKS: LazyLock<Mutex<HashMap<String, Arc<DownloadTask>>>> = LazyLock::new(Default::default);

#[wasm_bindgen(getter_with_clone)]
pub struct JsFileDownloadState {
    pub task_id: String,
    pub phase: String,
    pub downloaded_bytes: f64,
    pub total_bytes: f64,
    pub message: Option<String>,
}

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub async fn download(
    url: String,
    file_path_str: String,
    on_progress: js_sys::Function,
) -> Result<(), JsError> {
    download_internal(url, file_path_str, |state| {
        let _ = on_progress.call1(&JsValue::NULL, &JsValue::from(state));
    })
    .await
    .map_err(|err| JsError::from(err.as_ref()))
}

#[wasm_bindgen]
pub async fn pause(task_id: String) -> Result<(), JsError> {
    if let Some(task) = task(&task_id) {
        task.pause().await?;
    }
    Ok(())
}

#[wasm_bindgen]
pub async fn resume(task_id: String) -> Result<(), JsError> {
    if let Some(task) = task(&task_id) {
        task.download().await?;
    }
    Ok(())
}

fn task(task_id: &str) -> Option<Arc<DownloadTask>> {
    TASKS.lock().unwrap_or_else(PoisonError::into_inner).get(task_id).cloned()
}

async fn download_internal(
    url: String,
    file_path_str: String,
    callback: impl Fn(JsFileDownloadState),
) -> Result<(), Box<dyn Error>> {
    let request = DownloadTaskRequest::file().destination(PathBuf::from(file_path_str)).source_url(url).build();
    let task_id = request.download_id().to_string();
    let manager = MANAGER.get_or_init(|| async { DownloadManager::new(DownloadManagerType::default(), RuntimeHandle::current()) }).await;
    let task = manager.download_task(request).await?;
    task.delete().await?;
    TASKS.lock().unwrap_or_else(PoisonError::into_inner).insert(task_id.clone(), Arc::clone(&task));

    let mut progress = task.progress();
    task.download().await?;
    while let Some(state) = progress.next().await {
        let (phase, message) = match &state.phase {
            DownloadPhase::NotDownloaded {} => ("not_downloaded", None),
            DownloadPhase::Downloading {} => ("downloading", None),
            DownloadPhase::Paused {} => ("paused", None),
            DownloadPhase::Downloaded {} => ("downloaded", None),
            DownloadPhase::Locked {
                manager_id,
            } => ("locked", Some(manager_id.clone())),
            DownloadPhase::Error {
                message,
            } => ("error", Some(message.clone())),
        };
        callback(JsFileDownloadState {
            task_id: task_id.clone(),
            phase: phase.to_owned(),
            downloaded_bytes: state.downloaded_bytes as f64,
            total_bytes: state.total_bytes as f64,
            message,
        });
        match state.phase {
            DownloadPhase::Downloading {} => {
                printf!("Progress: {} / {} bytes", state.downloaded_bytes, state.total_bytes);
            },
            DownloadPhase::Downloaded {} => {
                printf!("Downloaded");
                break;
            },
            DownloadPhase::Error {
                message,
            } => {
                eprintf!("Error: {message}");
                return Err(DownloadError::Backend(message).into());
            },
            _ => {},
        }
    }
    Ok(())
}
