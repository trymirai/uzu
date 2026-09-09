use std::sync::{Arc, Mutex, PoisonError};

use futures_util::stream::{StreamExt, select_all};
use kiban::{rt, rt::TaskJoinHandle, stream::BoxStream};

use crate::{DownloadError, DownloadPhase, DownloadState, DownloadTask, DownloadTaskRequest, locks::LockError};

pub struct GroupDownloadTask {
    pub request: DownloadTaskRequest,
    children: Arc<[Arc<DownloadTask>]>,
    download_loop: Arc<Mutex<Option<Box<dyn TaskJoinHandle<()>>>>>,
}

impl GroupDownloadTask {
    pub fn new(
        request: DownloadTaskRequest,
        children: Vec<Arc<DownloadTask>>,
    ) -> Self {
        let group = Self {
            request,
            children: children.into(),
            download_loop: Arc::default(),
        };
        if group.state().phase == (DownloadPhase::Downloading {}) {
            group.start_download_loop();
        }
        group
    }

    pub fn subtasks(&self) -> &[Arc<DownloadTask>] {
        &self.children
    }

    pub fn state(&self) -> DownloadState {
        children_state(&self.children, &self.download_loop)
    }

    pub fn progress(&self) -> BoxStream<'static, DownloadState> {
        let children = Arc::clone(&self.children);
        let download_loop = Arc::clone(&self.download_loop);
        Box::pin(
            select_all(self.children.iter().map(|child| child.progress()))
                .map(move |_| children_state(&children, &download_loop)),
        )
    }

    pub async fn download(&self) -> Result<(), DownloadError> {
        self.stop_download_loop().await;
        let Some((current, state)) = self
            .children
            .iter()
            .map(|child| (child, child.state()))
            .find(|(_, state)| state.phase != (DownloadPhase::Downloaded {}))
        else {
            return Ok(());
        };
        if state.phase != (DownloadPhase::Downloading {}) {
            Box::pin(current.download()).await?;
        }
        self.start_download_loop();
        Ok(())
    }

    pub async fn pause(&self) -> Result<(), DownloadError> {
        self.stop_download_loop().await;
        let mut result = Ok(());
        for child in self.children.iter().filter(|child| child.state().is_in_progress()) {
            result = result.and(Box::pin(child.pause()).await);
        }
        result
    }

    pub async fn delete(&self) -> Result<(), DownloadError> {
        self.stop_download_loop().await;
        if let Some(manager_id) = self.foreign_owner().await {
            return Err(LockError::LockedByOther {
                manager_id,
            }
            .into());
        }
        let mut result = Ok(());
        for child in self.children.iter() {
            result = result.and(Box::pin(child.delete()).await);
        }
        result
    }

    pub async fn foreign_owner(&self) -> Option<String> {
        for child in self.children.iter() {
            if let Some(manager_id) = Box::pin(child.foreign_owner()).await {
                return Some(manager_id);
            }
        }
        None
    }

    fn start_download_loop(&self) {
        let children = Arc::clone(&self.children);
        let running = rt::spawn(async move { download_in_order(&children).await });
        if let Some(previous) = self.download_loop.lock().unwrap_or_else(PoisonError::into_inner).replace(running) {
            previous.abort();
        }
    }

    async fn stop_download_loop(&self) {
        let running = self.download_loop.lock().unwrap_or_else(PoisonError::into_inner).take();
        if let Some(running) = running {
            running.abort_and_join().await;
        }
    }
}

impl Drop for GroupDownloadTask {
    fn drop(&mut self) {
        if let Some(running) = self.download_loop.lock().unwrap_or_else(PoisonError::into_inner).take() {
            running.abort();
        }
    }
}

fn children_state(
    children: &[Arc<DownloadTask>],
    download_loop: &Mutex<Option<Box<dyn TaskJoinHandle<()>>>>,
) -> DownloadState {
    let download_loop_running = download_loop
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .as_ref()
        .is_some_and(|running| !running.is_finished());
    let mut total_bytes = 0;
    let mut downloaded_bytes = 0;
    let mut in_progress = None;
    let mut incomplete = None;
    for state in children.iter().map(|child| child.state()) {
        total_bytes += state.total_bytes;
        downloaded_bytes += state.downloaded_bytes;
        if state.phase.is_in_progress() {
            in_progress.get_or_insert(state.phase);
        } else if state.phase != (DownloadPhase::Downloaded {}) {
            incomplete.get_or_insert(state.phase);
        }
    }
    let phase = match (in_progress, incomplete) {
        (Some(phase), _) => phase,
        (None, None) => DownloadPhase::Downloaded {},
        (None, Some(DownloadPhase::NotDownloaded {} | DownloadPhase::Paused {})) if download_loop_running => {
            DownloadPhase::Downloading {}
        },
        (None, Some(DownloadPhase::NotDownloaded {})) if downloaded_bytes > 0 => DownloadPhase::Paused {},
        (None, Some(phase)) => phase,
    };
    DownloadState {
        total_bytes,
        downloaded_bytes,
        phase,
    }
}

async fn download_in_order(children: &[Arc<DownloadTask>]) {
    for child in children {
        let mut progress = child.progress();
        match child.state().phase {
            DownloadPhase::Downloaded {} => continue,
            DownloadPhase::Downloading {} => {},
            _ => {
                if child.download().await.is_err() {
                    return;
                }
            },
        }
        while let Some(state) = progress.next().await {
            match state.phase {
                DownloadPhase::Downloaded {} => break,
                phase if phase.is_in_progress() => {},
                _ => return,
            }
        }
    }
}
