use std::error::Error;

use tokio::{runtime::Handle, task::JoinHandle};

use crate::rt::TaskJoinHandle;

#[derive(Clone, Debug)]
pub struct TokioRuntimeHandle {
    handle: Handle,
}

impl TokioRuntimeHandle {
    pub fn current() -> Self {
        TokioRuntimeHandle {
            handle: Handle::current(),
        }
    }

    pub fn try_current() -> Result<Self, Box<dyn Error>> {
        Ok(TokioRuntimeHandle {
            handle: Handle::try_current()?,
        })
    }

    pub fn spawn<F: Future<Output: Send + 'static> + Send + 'static>(
        &self,
        future: F,
    ) -> Box<dyn TaskJoinHandle<F::Output>> {
        Box::new(self.handle.spawn(future))
    }
}

#[async_trait::async_trait]
impl<T: Send + 'static> TaskJoinHandle<T> for JoinHandle<T> {
    fn abort(&self) {
        self.abort();
    }

    async fn abort_and_join(self: Box<Self>) {
        self.abort();
        let _ = (*self).await;
    }
}

pub fn spawn<F: Future<Output: Send + 'static> + Send + 'static>(future: F) -> Box<dyn TaskJoinHandle<F::Output>> {
    Box::new(tokio::spawn(future))
}
