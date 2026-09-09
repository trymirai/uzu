use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use futures_util::stream::{StreamExt, select_all};
use kiban::stream::BoxStream;

use crate::{DownloadError, DownloadPhase, DownloadState, DownloadTask};

#[derive(Clone)]
pub struct DownloadTaskChildren(Arc<[Arc<DownloadTask>]>);

impl DownloadTaskChildren {
    pub fn new(children: Vec<Arc<DownloadTask>>) -> Self {
        Self(children.into())
    }

    pub fn as_slice(&self) -> &[Arc<DownloadTask>] {
        &self.0
    }

    pub fn state(
        &self,
        sequencing: bool,
    ) -> DownloadState {
        let mut total_bytes = 0;
        let mut downloaded_bytes = 0;
        let mut in_progress = None;
        let mut incomplete = None;
        for state in self.0.iter().map(|child| child.state()) {
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
            (None, Some(DownloadPhase::NotDownloaded {} | DownloadPhase::Paused {})) if sequencing => {
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

    pub fn progress(
        &self,
        sequencing: Arc<AtomicBool>,
    ) -> BoxStream<'static, DownloadState> {
        let children = self.clone();
        Box::pin(
            select_all(self.0.iter().map(|child| child.progress()))
                .map(move |_| children.state(sequencing.load(Ordering::Relaxed))),
        )
    }

    pub fn first_incomplete(&self) -> Option<(&Arc<DownloadTask>, DownloadState)> {
        self.0
            .iter()
            .map(|child| (child, child.state()))
            .find(|(_, state)| state.phase != (DownloadPhase::Downloaded {}))
    }

    pub async fn download_in_order(&self) {
        for child in self.0.iter() {
            let mut progress = child.progress();
            match child.state().phase {
                DownloadPhase::Downloaded {} => continue,
                phase if phase.is_in_progress() => {},
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

    pub async fn pause_downloading(&self) -> Result<(), DownloadError> {
        match self.0.iter().find(|child| child.state().can_pause()) {
            Some(child) => Box::pin(child.pause()).await,
            None => Ok(()),
        }
    }

    pub async fn foreign_owner(&self) -> Option<String> {
        for child in self.0.iter() {
            if let Some(manager_id) = Box::pin(child.foreign_owner()).await {
                return Some(manager_id);
            }
        }
        None
    }

    pub async fn delete_all(&self) -> Result<(), DownloadError> {
        let mut result = Ok(());
        for child in self.0.iter() {
            result = result.and(Box::pin(child.delete()).await);
        }
        result
    }
}
