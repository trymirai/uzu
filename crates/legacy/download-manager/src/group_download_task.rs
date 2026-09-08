use std::sync::{
    Arc, Mutex, PoisonError,
    atomic::{AtomicBool, Ordering},
};

use kiban::{rt, rt::TaskJoinHandle, stream::BoxStream};

use crate::{DownloadError, DownloadPhase, DownloadState, DownloadTask, DownloadTaskRequest, children::Children};

pub struct GroupDownloadTask {
    pub request: DownloadTaskRequest,
    children: Children,
    sequencing: Arc<AtomicBool>,
    driver: Mutex<Option<Box<dyn TaskJoinHandle<()>>>>,
}

impl GroupDownloadTask {
    pub fn new(
        request: DownloadTaskRequest,
        children: Vec<Arc<DownloadTask>>,
    ) -> Self {
        let group = Self {
            request,
            children: Children::new(children),
            sequencing: Arc::default(),
            driver: Mutex::default(),
        };
        if group.state().phase == (DownloadPhase::Downloading {}) {
            group.spawn_driver();
        }
        group
    }

    pub fn subtasks(&self) -> &[Arc<DownloadTask>] {
        self.children.as_slice()
    }

    pub fn state(&self) -> DownloadState {
        self.children.state(self.sequencing.load(Ordering::Relaxed))
    }

    pub fn progress(&self) -> BoxStream<'static, DownloadState> {
        self.children.progress(Arc::clone(&self.sequencing))
    }

    pub async fn download(&self) -> Result<(), DownloadError> {
        self.abort_driver().await;
        let Some((current, state)) = self.children.first_incomplete() else {
            return Ok(());
        };
        if !state.is_in_progress() {
            Box::pin(current.download()).await?;
        }
        self.spawn_driver();
        Ok(())
    }

    pub async fn pause(&self) -> Result<(), DownloadError> {
        self.abort_driver().await;
        self.children.pause_downloading().await
    }

    pub async fn delete(&self) -> Result<(), DownloadError> {
        self.abort_driver().await;
        if let Some(manager_id) = self.children.foreign_owner().await {
            return Err(DownloadError::LockedByOther(manager_id));
        }
        self.children.delete_all().await
    }

    pub async fn foreign_owner(&self) -> Option<String> {
        self.children.foreign_owner().await
    }

    fn spawn_driver(&self) {
        self.sequencing.store(true, Ordering::Relaxed);
        let children = self.children.clone();
        let sequencing = Arc::clone(&self.sequencing);
        let driver = rt::spawn(async move {
            children.download_in_order().await;
            sequencing.store(false, Ordering::Relaxed);
        });
        if let Some(previous) = self.driver.lock().unwrap_or_else(PoisonError::into_inner).replace(driver) {
            previous.abort();
        }
    }

    async fn abort_driver(&self) {
        let driver = self.driver.lock().unwrap_or_else(PoisonError::into_inner).take();
        if let Some(driver) = driver {
            driver.abort_and_join().await;
        }
        self.sequencing.store(false, Ordering::Relaxed);
    }
}

impl Drop for GroupDownloadTask {
    fn drop(&mut self) {
        if let Some(driver) = self.driver.lock().unwrap_or_else(PoisonError::into_inner).take() {
            driver.abort();
        }
    }
}
