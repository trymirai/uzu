use crate::backends::{apple::AppleBackend, common::DownloadManager};

pub type AppleDownloadManager = DownloadManager<AppleBackend>;
