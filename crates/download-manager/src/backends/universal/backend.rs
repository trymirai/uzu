use kiban::rt::RuntimeHandle;

use crate::{
    DownloadError,
    backends::{
        common,
        universal::{UniversalActiveTask, UniversalBackendContext, UniversalBackendError},
    },
    traits::DownloadBackend,
};

#[derive(Clone, Debug, Default)]
pub struct UniversalBackend;

impl DownloadBackend for UniversalBackend {
    type Context = UniversalBackendContext;
    type ActiveTask = UniversalActiveTask;
    type Error = UniversalBackendError;
}

impl common::Backend for UniversalBackend {
    const RESUME_ARTIFACT_EXTENSION: &'static str = "part";

    fn manager_suffix() -> &'static str {
        "universal"
    }

    fn create_context(runtime_handle: RuntimeHandle) -> Result<Self::Context, DownloadError> {
        Ok(UniversalBackendContext::new(runtime_handle))
    }
}
