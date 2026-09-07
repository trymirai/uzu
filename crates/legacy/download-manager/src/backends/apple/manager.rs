use crate::backends::{apple::AppleBackend, common::BackendDownloadManager};

pub type AppleDownloadManager = BackendDownloadManager<AppleBackend>;
