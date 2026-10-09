use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use download_manager::{DownloadManager, DownloadPhase, DownloadState, DownloadTask, DownloadTaskRequest};
use kiban::rt::TaskJoinHandle;
use shoji::types::model::ModelIdentifier;
use tokio::sync::{broadcast, watch};
use tokio_stream::StreamExt;

use super::StorageError;

pub type ModelTasks = HashMap<ModelIdentifier, Arc<ModelTask>>;
type TaskResult = Result<Arc<DownloadTask>, StorageError>;

pub struct ModelTask {
    pub request: Result<DownloadTaskRequest, StorageError>,
    total_bytes: i64,
    result: watch::Sender<Option<TaskResult>>,
    worker: Mutex<Option<Box<dyn TaskJoinHandle<()>>>>,
}

impl ModelTask {
    pub fn new(
        identifier: ModelIdentifier,
        request: Result<DownloadTaskRequest, StorageError>,
        total_bytes: i64,
        manager: Arc<DownloadManager>,
        events: broadcast::Sender<(ModelIdentifier, DownloadState)>,
        previous: Option<Arc<ModelTask>>,
    ) -> Self {
        let (result, _) = watch::channel(None);
        let sender = result.clone();
        let worker = kiban::rt::spawn({
            let request = request.clone();
            async move {
                if let Some(previous) = previous {
                    previous.stop().await;
                }
                let _ = events.send((
                    identifier.clone(),
                    DownloadState {
                        total_bytes,
                        downloaded_bytes: 0,
                        phase: DownloadPhase::Initializing {},
                    },
                ));
                let initialized = match request {
                    Ok(request) => tokio::select! {
                        result = manager.download_task(request) => result.map_err(StorageError::from),
                        _ = kiban::time::sleep(Duration::from_secs(10)) => Err(StorageError::DownloadManager {
                            message: "Timed out loading this model's download state. Retry the download to try again.".to_string(),
                        }),
                    },
                    Err(error) => Err(error),
                };
                sender.send_replace(Some(initialized.clone()));
                match initialized {
                    Ok(task) => {
                        let mut progress = task.progress();
                        while let Some(state) = progress.next().await {
                            let _ = events.send((identifier.clone(), state));
                        }
                    },
                    Err(error) => {
                        tracing::warn!(%identifier, %error, "unable to initialize model download");
                        let _ = events.send((identifier, Self::error_state(total_bytes, &error)));
                    },
                }
            }
        });
        Self {
            request,
            total_bytes,
            result,
            worker: Mutex::new(Some(worker)),
        }
    }

    pub fn state(&self) -> DownloadState {
        match self.result.borrow().as_ref() {
            Some(Ok(task)) => task.state(),
            Some(Err(error)) => Self::error_state(self.total_bytes, error),
            None => DownloadState {
                total_bytes: self.total_bytes,
                downloaded_bytes: 0,
                phase: DownloadPhase::Initializing {},
            },
        }
    }

    pub fn failed(&self) -> bool {
        matches!(self.result.borrow().as_ref(), Some(Err(_)))
    }

    pub async fn ready(&self) -> TaskResult {
        let mut result = self.result.subscribe();
        loop {
            if let Some(result) = result.borrow().clone() {
                return result;
            }
            result.changed().await.map_err(|_| StorageError::DownloadManager {
                message: "Model download initialization stopped".to_string(),
            })?;
        }
    }

    async fn stop(&self) {
        let worker = self.worker.lock().expect("model worker mutex poisoned").take();
        if let Some(worker) = worker {
            worker.abort_and_join().await;
        }
        self.result.send_replace(Some(Err(StorageError::DownloadManager {
            message: "Model download state was replaced by a newer catalog entry".to_string(),
        })));
    }

    fn error_state(
        total_bytes: i64,
        error: &StorageError,
    ) -> DownloadState {
        DownloadState {
            total_bytes,
            downloaded_bytes: 0,
            phase: DownloadPhase::Error {
                message: error.to_string(),
            },
        }
    }
}

impl Drop for ModelTask {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.get_mut().expect("model worker mutex poisoned").take() {
            worker.abort();
        }
    }
}
