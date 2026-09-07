use std::sync::{Arc, Mutex, PoisonError};

use futures_util::stream::select_all;
use kiban::{
    future::BoxFuture,
    rt::{self, TaskJoinHandle},
};
use tokio::sync::broadcast::{Sender as TokioBroadcastSender, channel as tokio_broadcast_channel};
use tokio_stream::{StreamExt, wrappers::BroadcastStream};

use crate::{DownloadError, DownloadPhase, DownloadState, DownloadTask, DownloadTaskRequest};

pub struct GroupDownloadTask {
    pub request: DownloadTaskRequest,
    children: Arc<[Arc<DownloadTask>]>,
    progress: TokioBroadcastSender<DownloadState>,
    driver: Mutex<Option<Box<dyn TaskJoinHandle<()>>>>,
}

impl GroupDownloadTask {
    pub fn new(
        request: DownloadTaskRequest,
        children: Vec<Arc<DownloadTask>>,
    ) -> Self {
        let (progress, _) = tokio_broadcast_channel(64);
        let group = Self {
            request,
            children: children.into(),
            progress,
            driver: Mutex::new(None),
        };
        if group.state().is_in_progress() {
            group.spawn_driver();
        }
        group
    }

    pub fn subtasks(&self) -> &[Arc<DownloadTask>] {
        &self.children
    }

    pub fn state(&self) -> DownloadState {
        group_state(&self.children)
    }

    pub fn progress(&self) -> BroadcastStream<DownloadState> {
        BroadcastStream::new(self.progress.subscribe())
    }

    pub fn download(&self) -> BoxFuture<'_, Result<(), DownloadError>> {
        Box::pin(async move {
            self.abort_driver().await;
            let result = match current_child(&self.children) {
                Some(current) => current.download().await,
                None => Ok(()),
            };
            self.emit();
            if result.is_ok() {
                self.spawn_driver();
            }
            result
        })
    }

    pub fn pause(&self) -> BoxFuture<'_, Result<(), DownloadError>> {
        Box::pin(async move {
            self.abort_driver().await;
            let result = match current_child(&self.children) {
                Some(current) => match current.pause().await {
                    Ok(()) | Err(DownloadError::InvalidStateTransition) => Ok(()),
                    Err(error) => Err(error),
                },
                None => Ok(()),
            };
            self.emit();
            result
        })
    }

    pub fn delete(&self) -> BoxFuture<'_, Result<(), DownloadError>> {
        Box::pin(async move {
            self.abort_driver().await;
            if let Some(owner) = self.foreign_lock().await {
                return Err(DownloadError::LockedByOther(owner));
            }
            let mut result = Ok(());
            for child in self.children.iter() {
                if let Err(error) = child.delete().await
                    && result.is_ok()
                {
                    result = Err(error);
                }
            }
            self.emit();
            result
        })
    }

    pub fn foreign_lock(&self) -> BoxFuture<'_, Option<String>> {
        Box::pin(async move {
            for child in self.children.iter() {
                if let Some(owner) = child.foreign_lock().await {
                    return Some(owner);
                }
            }
            None
        })
    }

    fn emit(&self) {
        let _ = self.progress.send(self.state());
    }

    fn spawn_driver(&self) {
        let driver = rt::spawn(run(Arc::clone(&self.children), self.progress.clone()));
        *self.driver.lock().unwrap_or_else(PoisonError::into_inner) = Some(driver);
    }

    async fn abort_driver(&self) {
        let driver = self.driver.lock().unwrap_or_else(PoisonError::into_inner).take();
        if let Some(driver) = driver {
            driver.abort_and_join().await;
        }
    }
}

impl Drop for GroupDownloadTask {
    fn drop(&mut self) {
        if let Some(driver) = self.driver.get_mut().unwrap_or_else(PoisonError::into_inner).take() {
            driver.abort();
        }
    }
}

async fn run(
    children: Arc<[Arc<DownloadTask>]>,
    progress: TokioBroadcastSender<DownloadState>,
) {
    let emit = || {
        let _ = progress.send(group_state(&children));
    };
    while let Some(current) = current_child(&children) {
        let mut events = select_all(children.iter().map(|child| child.progress()));
        match current.state().phase {
            DownloadPhase::Downloading {} => {},
            DownloadPhase::Downloaded {} => continue,
            _ => {
                if current.download().await.is_err() {
                    break;
                }
            },
        }
        emit();
        loop {
            if events.next().await.is_none() {
                emit();
                return;
            }
            match current.state().phase {
                DownloadPhase::Downloading {} => emit(),
                DownloadPhase::Downloaded {} => break,
                _ => {
                    emit();
                    return;
                },
            }
        }
    }
    emit();
}

fn current_child(children: &[Arc<DownloadTask>]) -> Option<&Arc<DownloadTask>> {
    let mut first_pending = None;
    for child in children {
        match child.state().phase {
            DownloadPhase::Downloading {} => return Some(child),
            DownloadPhase::Downloaded {} => {},
            _ => first_pending = first_pending.or(Some(child)),
        }
    }
    first_pending
}

fn group_state(children: &[Arc<DownloadTask>]) -> DownloadState {
    let mut total_bytes = 0;
    let mut downloaded_bytes = 0;
    let mut first_pending = None;
    let mut any_downloading = false;
    for child in children {
        let state = child.state();
        total_bytes += state.total_bytes;
        downloaded_bytes += state.downloaded_bytes;
        match state.phase {
            DownloadPhase::Downloading {} => any_downloading = true,
            DownloadPhase::Downloaded {} => {},
            phase => first_pending = first_pending.or(Some(phase)),
        }
    }
    let phase = if any_downloading {
        DownloadPhase::Downloading {}
    } else {
        match first_pending {
            None => DownloadPhase::Downloaded {},
            Some(DownloadPhase::NotDownloaded {}) if downloaded_bytes > 0 => DownloadPhase::Paused {},
            Some(phase) => phase,
        }
    };
    DownloadState {
        total_bytes,
        downloaded_bytes,
        phase,
    }
}
