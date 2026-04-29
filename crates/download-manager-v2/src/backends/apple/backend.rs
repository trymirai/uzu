use crate::{
    backends::apple::{AppleActiveTask, AppleBackendContext, AppleBackendError},
    traits::DownloadBackend,
};

#[derive(Clone, Debug, Default)]
pub struct AppleBackend;

impl DownloadBackend for AppleBackend {
    type Context = AppleBackendContext;
    type ActiveTask = AppleActiveTask;
    type Error = AppleBackendError;
}
