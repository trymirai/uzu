use std::error::Error;

use futures_util::future::{AbortHandle, Abortable};
use tokio::sync::oneshot;

use crate::rt::TaskJoinHandle;

#[derive(Clone, Debug)]
pub struct WasmRuntimeHandle;

impl WasmRuntimeHandle {
    pub fn current() -> Self {
        WasmRuntimeHandle
    }

    pub fn try_current() -> Result<Self, Box<dyn Error>> {
        Ok(WasmRuntimeHandle)
    }

    pub fn spawn<F: Future<Output: 'static> + 'static>(
        &self,
        future: F,
    ) -> Box<dyn TaskJoinHandle<F::Output>> {
        spawn(future)
    }
}

pub struct WasmTaskJoinHandle {
    abort_handle: AbortHandle,
    completion: oneshot::Receiver<()>,
}

#[async_trait::async_trait(?Send)]
impl<T> TaskJoinHandle<T> for WasmTaskJoinHandle {
    fn abort(&self) {
        self.abort_handle.abort();
    }

    async fn abort_and_join(self: Box<Self>) {
        self.abort_handle.abort();
        let _ = self.completion.await;
    }
}

pub fn spawn<F: Future<Output: 'static> + 'static>(future: F) -> Box<dyn TaskJoinHandle<F::Output>> {
    let (abort_handle, abort_registration) = AbortHandle::new_pair();
    let future = Abortable::new(future, abort_registration);
    let (completion_sender, completion) = oneshot::channel();
    wasm_bindgen_futures::spawn_local(async move {
        let _ = future.await;
        let _ = completion_sender.send(());
    });
    Box::new(WasmTaskJoinHandle {
        abort_handle,
        completion,
    })
}
