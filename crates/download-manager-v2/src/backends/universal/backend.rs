use crate::{
    backends::universal::{UniversalActiveTask, UniversalBackendContext, UniversalBackendError},
    traits::DownloadBackend,
};

#[derive(Clone, Debug, Default)]
pub struct UniversalBackend;

impl DownloadBackend for UniversalBackend {
    type Context = UniversalBackendContext;
    type ActiveTask = UniversalActiveTask;
    type Error = UniversalBackendError;
}
