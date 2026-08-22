use std::{
    collections::HashMap,
    error::Error,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock},
};

use download_manager::{
    DownloadError, FileCheck, FileDownloadGroup, FileDownloadGroupPhase, FileDownloadGroupSpec, FileDownloadManager,
    FileDownloadRequest, RelativeFilePath, compute_download_id,
};
use kiban::{eprintf, fs, printf, rt::RuntimeHandle};
use tokio::sync::{Mutex, OnceCell};
use tokio_stream::StreamExt;
use wasm_bindgen::{JsError, JsValue, prelude::wasm_bindgen};

static MANAGER: OnceCell<Arc<dyn FileDownloadManager>> = OnceCell::const_new();
static GROUPS: LazyLock<Mutex<HashMap<String, FileDownloadGroup>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

#[wasm_bindgen]
pub struct JsFileDownloadState {
    #[wasm_bindgen(getter_with_clone)]
    pub task_id: String,
    #[wasm_bindgen(getter_with_clone)]
    pub phase: String,
    pub downloaded_bytes: f64,
    pub total_bytes: f64,
    #[wasm_bindgen(getter_with_clone)]
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
    let group = GROUPS.lock().await.get(&task_id).cloned();
    if let Some(group) = group {
        group.pause().await?;
    }

    Ok(())
}

#[wasm_bindgen]
pub async fn resume(task_id: String) -> Result<(), JsError> {
    let group = GROUPS.lock().await.get(&task_id).cloned();
    if let Some(group) = group {
        group.download().await?;
    }

    Ok(())
}

async fn download_internal(
    url: String,
    file_path_str: String,
    callback: impl Fn(JsFileDownloadState),
) -> Result<(), Box<dyn Error>> {
    let file_path = PathBuf::from(file_path_str);
    let manager = get_manager().await?;
    let task_id = compute_download_id(&file_path).to_string();

    if let Some(previous_group) = GROUPS.lock().await.remove(&task_id) {
        previous_group.cancel().await?;
    } else if fs::asyn::try_exists(&file_path).await? {
        fs::asyn::remove_file(&file_path).await?;
    }

    let destination_root = file_path.parent().filter(|path| !path.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let relative_path = RelativeFilePath::try_from(
        file_path.file_name().ok_or("download path must name a file")?.to_string_lossy().as_ref(),
    )?;
    let spec = FileDownloadGroupSpec::new(
        destination_root,
        [FileDownloadRequest::new(url, relative_path, FileCheck::None, None)],
    )?;
    let group = FileDownloadGroup::open(manager, spec).await?;
    GROUPS.lock().await.insert(task_id.clone(), group.clone());
    let mut progress_stream = group.subscribe();
    let attempt = group.download().await?;

    let mut download_error = None;
    while let Some(state) = progress_stream.next().await {
        let failure_message = || {
            state
                .failures
                .iter()
                .map(|failure| format!("{}: {}", failure.relative_path, failure.error))
                .collect::<Vec<_>>()
                .join("; ")
        };
        let (phase, message) = match state.phase {
            FileDownloadGroupPhase::NotDownloaded => ("not_downloaded", None),
            FileDownloadGroupPhase::Downloading => ("downloading", None),
            FileDownloadGroupPhase::Paused => ("paused", None),
            FileDownloadGroupPhase::Downloaded => ("downloaded", None),
            FileDownloadGroupPhase::Locked => ("locked", Some(failure_message())),
            FileDownloadGroupPhase::Error => ("error", Some(failure_message())),
        };
        let js_state = JsFileDownloadState {
            task_id: task_id.clone(),
            phase: phase.to_owned(),
            downloaded_bytes: state.downloaded_bytes as f64,
            total_bytes: state.total_bytes.unwrap_or(0) as f64,
            message: message.clone(),
        };
        callback(js_state);

        match state.phase {
            FileDownloadGroupPhase::Downloading => {
                printf!("Progress: {} / {:?} bytes ({:?})", state.downloaded_bytes, state.total_bytes, state.phase);
            },
            FileDownloadGroupPhase::Downloaded => {
                printf!("Downloaded state");
                break;
            },
            FileDownloadGroupPhase::Error => {
                let error = message.unwrap_or_else(|| "download failed".to_owned());
                eprintf!("Error: {error}");
                download_error = Some(error);
                break;
            },
            _ => (),
        }
    }
    attempt.wait().await?;

    if let Some(err) = download_error {
        return Err(DownloadError::Backend(err).into());
    }

    let final_state = group.state();
    if matches!(final_state.phase, FileDownloadGroupPhase::Error | FileDownloadGroupPhase::Locked) {
        let message = final_state
            .failures
            .iter()
            .map(|failure| format!("{}: {}", failure.relative_path, failure.error))
            .collect::<Vec<_>>()
            .join("; ");
        return Err(DownloadError::Backend(message).into());
    }

    Ok(())
}

async fn get_manager() -> Result<Arc<dyn FileDownloadManager>, DownloadError> {
    MANAGER
        .get_or_try_init(|| async {
            <dyn FileDownloadManager>::system_default(RuntimeHandle::current()).await.map(Arc::from)
        })
        .await
        .map(Arc::clone)
}
