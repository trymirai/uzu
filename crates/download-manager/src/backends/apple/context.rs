use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, Mutex},
};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{
    NSData, NSString, NSURL, NSURLSession, NSURLSessionConfiguration, NSURLSessionDelegate, NSURLSessionDownloadTask,
};
use tokio::runtime::Handle as TokioHandle;

use crate::{
    DownloadInfo, FileCheck,
    backends::apple::{
        AppleActiveTask, AppleBackend, AppleBackendError, AppleEventRegistry, AppleEventSink, AppleGetTasksHandler,
        AppleSessionDelegate, task_ext::AppleDownloadTaskExt,
    },
    traits::{ActiveDownloadGeneration, BackendContext, BackendEventSender, DownloadConfig},
};

pub struct AppleBackendContext {
    session: Retained<NSURLSession>,
    _delegate: Retained<AppleSessionDelegate>,
    _delegate_protocol_object: Retained<ProtocolObject<dyn NSURLSessionDelegate>>,
    event_registry: AppleEventRegistry,
    tokio_handle: TokioHandle,
}

impl std::fmt::Debug for AppleBackendContext {
    fn fmt(
        &self,
        formatter: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        formatter.debug_struct("AppleBackendContext").finish_non_exhaustive()
    }
}

impl Default for AppleBackendContext {
    fn default() -> Self {
        Self::new(TokioHandle::current())
    }
}

impl AppleBackendContext {
    pub fn new(tokio_handle: TokioHandle) -> Self {
        let event_registry = Arc::new(Mutex::new(HashMap::new()));
        let delegate = AppleSessionDelegate::new(Arc::clone(&event_registry));
        let delegate_protocol_object = AppleSessionDelegate::protocol_object(delegate.clone());
        let session = unsafe {
            NSURLSession::sessionWithConfiguration_delegate_delegateQueue(
                &NSURLSessionConfiguration::ephemeralSessionConfiguration(),
                Some(&delegate_protocol_object),
                None,
            )
        };

        Self {
            session,
            _delegate: delegate,
            _delegate_protocol_object: delegate_protocol_object,
            event_registry,
            tokio_handle,
        }
    }

    pub fn matching_download_task(
        &self,
        config: &DownloadConfig,
    ) -> Option<objc2::rc::Retained<NSURLSessionDownloadTask>> {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let sender = Arc::new(Mutex::new(Some(sender)));
        let handler = AppleGetTasksHandler::new({
            let sender = Arc::clone(&sender);
            move |_data_tasks, _upload_tasks, download_tasks| {
                if let Some(sender) = sender.lock().ok().and_then(|mut sender| sender.take()) {
                    let _ = sender.send(download_tasks);
                }
            }
        });
        unsafe {
            self.session.getTasksWithCompletionHandler(&handler);
        }
        let download_tasks = receiver.recv().unwrap_or_default();
        download_tasks.into_iter().find(|task| task.download_id() == Some(config.download_id))
    }

    pub fn attach_existing_task(
        &self,
        task: &NSURLSessionDownloadTask,
        config: Arc<DownloadConfig>,
        generation: ActiveDownloadGeneration,
        backend_event_sender: BackendEventSender,
    ) {
        self.prepare_task(task, config, generation, backend_event_sender);
    }
}

unsafe impl Send for AppleBackendContext {}
unsafe impl Sync for AppleBackendContext {}

#[async_trait::async_trait]
impl BackendContext for AppleBackendContext {
    type Backend = AppleBackend;

    async fn download(
        &self,
        config: Arc<DownloadConfig>,
        generation: ActiveDownloadGeneration,
        backend_event_sender: BackendEventSender,
    ) -> Result<AppleActiveTask, AppleBackendError> {
        let ns_url = NSURL::URLWithString(&NSString::from_str(&config.source_url)).ok_or(AppleBackendError::BadUrl)?;
        let task = self.session.downloadTaskWithURL(&ns_url);
        self.prepare_task(&task, config, generation, backend_event_sender);
        task.resume();
        Ok(AppleActiveTask::wrap(task))
    }

    async fn resume(
        &self,
        config: Arc<DownloadConfig>,
        generation: ActiveDownloadGeneration,
        resume_artifact_path: &Path,
        backend_event_sender: BackendEventSender,
    ) -> Result<AppleActiveTask, AppleBackendError> {
        let resume_data =
            tokio::fs::read(resume_artifact_path).await.map_err(|error| AppleBackendError::Io(error.to_string()))?;
        let ns_data = NSData::with_bytes(&resume_data);
        let task = self.session.downloadTaskWithResumeData(&ns_data);
        self.prepare_task(&task, config, generation, backend_event_sender);
        task.resume();
        Ok(AppleActiveTask::wrap(task))
    }
}

impl AppleBackendContext {
    fn prepare_task(
        &self,
        task: &NSURLSessionDownloadTask,
        config: Arc<DownloadConfig>,
        generation: ActiveDownloadGeneration,
        backend_event_sender: BackendEventSender,
    ) {
        let download_info = match &config.file_check {
            FileCheck::CRC(crc) => DownloadInfo::with_crc(
                config.source_url.clone(),
                config.destination.to_string_lossy().to_string(),
                crc.clone(),
            ),
            FileCheck::None => {
                DownloadInfo::new(config.source_url.clone(), config.destination.to_string_lossy().to_string())
            },
        };
        task.set_download_info(&download_info);
        if let Ok(mut registry) = self.event_registry.lock() {
            registry.insert(
                config.download_id,
                AppleEventSink {
                    generation,
                    destination: config.destination.clone(),
                    backend_event_sender,
                    tokio_handle: self.tokio_handle.clone(),
                },
            );
        }
    }
}
