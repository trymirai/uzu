use std::path::PathBuf;

use kiban::rt::RuntimeHandle;

use crate::backends::{BackendEventSender, DownloadGeneration};

#[derive(Clone, Debug)]
pub struct AppleEventSink {
    pub generation: DownloadGeneration,
    pub destination: PathBuf,
    pub expected_bytes: Option<u64>,
    pub events: BackendEventSender,
    pub runtime_handle: RuntimeHandle,
}
