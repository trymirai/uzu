use crate::backends::{common::BackendDownloadManager, universal::UniversalBackend};

pub type UniversalDownloadManager = BackendDownloadManager<UniversalBackend>;
