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

    fn create_context(_tokio_handle: tokio::runtime::Handle) -> Result<Self::Context, DownloadError> {
        Ok(UniversalBackendContext::default())
    }
}
