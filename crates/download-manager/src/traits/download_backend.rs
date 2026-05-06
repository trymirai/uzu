use std::fmt::Debug;

use crate::traits::{ActiveTask, BackendContext};

pub trait DownloadBackend: Debug + Clone + Send + Sync + 'static {
    type Context: BackendContext<Backend = Self>;
    type ActiveTask: ActiveTask<Backend = Self>;
    type Error: std::error::Error + Send + Sync + 'static;
}
