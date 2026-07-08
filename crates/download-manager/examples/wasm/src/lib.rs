mod progress;

use std::{error::Error, path::PathBuf};

use download_manager::{DownloadError, FileCheck, FileDownloadManager, FileDownloadPhase};
use kiban::{eprintf, fs, printf, rt::RuntimeHandle};
use tokio::sync::OnceCell;
use tokio_stream::StreamExt;
use wasm_bindgen::{JsError, JsValue, prelude::wasm_bindgen};

use crate::progress::JsFileDownloadState;

static MANAGER: OnceCell<Box<dyn FileDownloadManager>> = OnceCell::const_new();

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
    let manager = get_manager().await?;
    let all_tasks = manager.get_all_file_tasks().await?;
    for task in all_tasks {
        if task.download_id().to_string() == task_id {
            task.pause().await?;
        }
    }

    Ok(())
}

#[wasm_bindgen]
pub async fn resume(task_id: String) -> Result<(), JsError> {
    let manager = get_manager().await?;
    let all_tasks = manager.get_all_file_tasks().await?;
    for task in all_tasks {
        if task.download_id().to_string() == task_id {
            task.download().await?;
        }
    }

    Ok(())
}

async fn download_internal(
    url: String,
    file_path_str: String,
    callback: impl Fn(JsFileDownloadState),
) -> Result<(), Box<dyn Error>> {
    let file_path = PathBuf::from(file_path_str);
    if fs::asyn::try_exists(&file_path).await? {
        fs::asyn::remove_file(&file_path).await?;
    }

    let manager = get_manager().await?;
    let task = manager.file_download_task(&url, &file_path, FileCheck::None, None).await?;
    let mut progress_stream = task.progress().await?;
    task.download().await?;

    while let Some(Ok(state)) = progress_stream.next().await {
        let (phase, message) = match &state.phase {
            FileDownloadPhase::NotDownloaded => ("not_downloaded", None),
            FileDownloadPhase::Downloading => ("downloading", None),
            FileDownloadPhase::Paused => ("paused", None),
            FileDownloadPhase::Downloaded => ("downloaded", None),
            FileDownloadPhase::LockedByOther(id) => ("locked", Some(id.clone())),
            FileDownloadPhase::Error(err) => ("error", Some(err.clone())),
        };
        let js_state = JsFileDownloadState {
            task_id: task.download_id().to_string(),
            phase: phase.to_owned(),
            downloaded_bytes: state.downloaded_bytes as f64,
            total_bytes: state.total_bytes as f64,
            message,
        };
        callback(js_state);

        match state.phase {
            FileDownloadPhase::Downloading => {
                printf!("Progress: {} / {} bytes ({:?})", state.downloaded_bytes, state.total_bytes, state.phase);
            },
            FileDownloadPhase::Downloaded => {
                printf!("Downloaded state");
                break;
            },
            FileDownloadPhase::Error(err) => {
                eprintf!("Error: {err}");
                break;
            },
            _ => (),
        }
    }
    task.wait().await;

    Ok(())
}

async fn get_manager() -> Result<&'static dyn FileDownloadManager, DownloadError> {
    MANAGER
        .get_or_try_init(|| <dyn FileDownloadManager>::system_default(RuntimeHandle::current()))
        .await
        .map(Box::as_ref)
}
