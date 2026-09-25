use std::{
    path::Path,
    sync::{Mutex, PoisonError},
};

use block2::RcBlock;
use kiban::fs;
use objc2::rc::Retained;
use objc2_foundation::{NSData, NSURLSessionDownloadTask};
use tokio::sync::oneshot::channel as tokio_oneshot_channel;

use crate::backends::{
    ActiveTask, BackendError,
    apple::{AppleBackendError, AppleEventRegistry},
};

pub struct AppleActiveTask {
    task: Retained<NSURLSessionDownloadTask>,
    event_registry: AppleEventRegistry,
    authenticated: bool,
}

impl AppleActiveTask {
    pub fn new(
        task: Retained<NSURLSessionDownloadTask>,
        event_registry: AppleEventRegistry,
        authenticated: bool,
    ) -> Self {
        Self {
            task,
            event_registry,
            authenticated,
        }
    }

    fn unregister(&self) {
        self.event_registry.lock().unwrap_or_else(PoisonError::into_inner).remove(&self.task.taskIdentifier());
    }
}

#[async_trait::async_trait]
impl ActiveTask for AppleActiveTask {
    async fn pause(
        self: Box<Self>,
        resume_artifact_path: &Path,
    ) -> Result<(), BackendError> {
        // The resume blob archives the original and current requests with all their headers, so producing one
        // for an authenticated task would write the bearer token to disk and freeze it into the resumed request.
        // Authenticated downloads therefore start over after a pause or a relaunch on this backend; the universal
        // backend resumes them from the partial file and re-sends the token from memory instead.
        if self.authenticated {
            self.cancel().await;
            return Ok(());
        }
        self.unregister();
        let (sender, receiver) = tokio_oneshot_channel::<Vec<u8>>();
        {
            let sender = Mutex::new(Some(sender));
            let handler = RcBlock::new(move |resume_data: *mut NSData| {
                if let Some(sender) = sender.lock().unwrap_or_else(PoisonError::into_inner).take() {
                    let _ = sender.send(unsafe { resume_data.as_ref() }.map(|data| data.to_vec()).unwrap_or_default());
                }
            });
            unsafe {
                self.task.cancelByProducingResumeData(&handler);
            }
        }
        let resume_data = receiver.await.map_err(AppleBackendError::CallbackDropped)?;
        fs::asyn::write(resume_artifact_path, resume_data).await?;
        Ok(())
    }

    async fn cancel(self: Box<Self>) {
        self.unregister();
        self.task.cancel();
    }
}
