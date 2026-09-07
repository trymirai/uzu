use std::path::PathBuf;

use kiban::rt::RuntimeHandle;

use crate::backends::common::{ActiveDownloadGeneration, BackendEventSender};

#[derive(Clone, Debug)]
pub struct AppleEventSink {
    pub generation: ActiveDownloadGeneration,
    pub destination: PathBuf,
    pub backend_event_sender: BackendEventSender,
    pub runtime_handle: RuntimeHandle,
}
