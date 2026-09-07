use std::sync::Arc;

use tokio_stream::wrappers::BroadcastStream;

use crate::{DownloadError, DownloadState, DownloadTaskRequest, FileDownloadTask, GroupDownloadTask};

pub enum DownloadTask {
    File(FileDownloadTask),
    Group(GroupDownloadTask),
}

impl DownloadTask {
    pub fn request(&self) -> &DownloadTaskRequest {
        match self {
            Self::File(file) => &file.request,
            Self::Group(group) => &group.request,
        }
    }

    pub fn subtasks(&self) -> &[Arc<DownloadTask>] {
        match self {
            Self::File(_) => &[],
            Self::Group(group) => group.subtasks(),
        }
    }

    pub fn state(&self) -> DownloadState {
        match self {
            Self::File(file) => file.state(),
            Self::Group(group) => group.state(),
        }
    }

    pub fn progress(&self) -> BroadcastStream<DownloadState> {
        match self {
            Self::File(file) => file.progress(),
            Self::Group(group) => group.progress(),
        }
    }

    pub async fn download(&self) -> Result<(), DownloadError> {
        match self {
            Self::File(file) => file.download().await,
            Self::Group(group) => group.download().await,
        }
    }

    pub async fn pause(&self) -> Result<(), DownloadError> {
        match self {
            Self::File(file) => file.pause().await,
            Self::Group(group) => group.pause().await,
        }
    }

    pub async fn delete(&self) -> Result<(), DownloadError> {
        match self {
            Self::File(file) => file.delete().await,
            Self::Group(group) => group.delete().await,
        }
    }

    pub async fn foreign_lock(&self) -> Option<String> {
        match self {
            Self::File(file) => file.foreign_lock().await,
            Self::Group(group) => group.foreign_lock().await,
        }
    }
}
