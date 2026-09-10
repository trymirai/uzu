use crate::backends::DownloadGeneration;

#[derive(Clone, Copy, Debug)]
pub struct BackendProgress {
    pub generation: DownloadGeneration,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
}
