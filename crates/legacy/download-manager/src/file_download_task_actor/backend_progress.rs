use crate::traits::ActiveDownloadGeneration;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendProgress {
    pub generation: ActiveDownloadGeneration,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
}
