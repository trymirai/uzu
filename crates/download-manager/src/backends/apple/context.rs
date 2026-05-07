use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, Mutex},
};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{
    NSBundle, NSData, NSString, NSURL, NSURLSession, NSURLSessionConfiguration, NSURLSessionDelegate,
    NSURLSessionDownloadTask, NSURLSessionTaskState,
};
use tokio::{runtime::Handle as TokioHandle, sync::oneshot::channel as tokio_oneshot_channel};

use crate::{
    DownloadInfo, FileCheck,
    backends::apple::{
        AppleActiveTask, AppleBackend, AppleBackendError, AppleEventRegistry, AppleEventSink, AppleGetTasksHandler,
        AppleSessionDelegate, task_ext::AppleDownloadTaskExt,
    },
    lock_manager::DestinationLockLease,
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

impl AppleBackendContext {
    pub fn new(tokio_handle: TokioHandle) -> Self {
        let event_registry = Arc::new(Mutex::new(HashMap::new()));
        let delegate = AppleSessionDelegate::new(Arc::clone(&event_registry));
        let delegate_protocol_object = AppleSessionDelegate::protocol_object(delegate.clone());
        let session = unsafe {
            NSURLSession::sessionWithConfiguration_delegate_delegateQueue(
                &automatic_session_configuration(),
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

    pub(crate) async fn claim_matching_download_task(
        &self,
        config: &DownloadConfig,
    ) -> Result<Option<Retained<NSURLSessionDownloadTask>>, AppleBackendError> {
        self.find_download_task(config).await
    }

    pub(crate) async fn has_download_task_to_claim(
        &self,
        config: &DownloadConfig,
    ) -> Result<bool, AppleBackendError> {
        let download_tasks = self.download_tasks().await?;
        Ok(download_tasks
            .iter()
            .any(|task| task.download_id() == Some(config.download_id) && is_live_task_state(task.state())))
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn event_sink_count_for_download(
        &self,
        download_id: crate::DownloadId,
    ) -> usize {
        self.event_registry
            .lock()
            .map(|registry| registry.keys().filter(|(id, _)| *id == download_id).count())
            .unwrap_or(0)
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn event_sink_task_identifiers_for_download(
        &self,
        download_id: crate::DownloadId,
    ) -> Vec<u64> {
        self.event_registry
            .lock()
            .map(|registry| {
                registry
                    .keys()
                    .filter_map(|(id, task_identifier)| (*id == download_id).then_some(*task_identifier))
                    .collect()
            })
            .unwrap_or_default()
    }

    async fn find_download_task(
        &self,
        config: &DownloadConfig,
    ) -> Result<Option<Retained<NSURLSessionDownloadTask>>, AppleBackendError> {
        let download_tasks = self.download_tasks().await?;
        Ok(select_matching_download_task(download_tasks, config))
    }

    async fn download_tasks(&self) -> Result<Box<[Retained<NSURLSessionDownloadTask>]>, AppleBackendError> {
        let (download_tasks_sender, download_tasks_receiver) = tokio_oneshot_channel();
        let pending_download_tasks_sender = Arc::new(Mutex::new(Some(download_tasks_sender)));
        let handler = AppleGetTasksHandler::new({
            let pending_download_tasks_sender = Arc::clone(&pending_download_tasks_sender);
            move |_data_tasks, _upload_tasks, download_tasks| {
                let download_tasks_sender = match pending_download_tasks_sender.lock() {
                    Ok(mut download_tasks_sender) => download_tasks_sender.take(),
                    Err(poisoned_sender) => {
                        let mut download_tasks_sender = poisoned_sender.into_inner();
                        download_tasks_sender.take()
                    },
                };
                if let Some(download_tasks_sender) = download_tasks_sender {
                    let _ = download_tasks_sender.send(download_tasks);
                }
            }
        });
        unsafe {
            self.session.getTasksWithCompletionHandler(&handler);
        }
        download_tasks_receiver.await.map_err(|error| {
            AppleBackendError::TaskEnumeration(format!("URLSession task enumeration callback dropped: {error}"))
        })
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

    pub(crate) fn event_registry(&self) -> AppleEventRegistry {
        Arc::clone(&self.event_registry)
    }
}

fn select_matching_download_task(
    download_tasks: Box<[Retained<NSURLSessionDownloadTask>]>,
    config: &DownloadConfig,
) -> Option<Retained<NSURLSessionDownloadTask>> {
    let mut live_match = None;
    for task in download_tasks {
        if task.download_id() != Some(config.download_id) {
            continue;
        }
        let task_state = task.state();
        if !is_live_task_state(task_state) {
            continue;
        }
        let info = task.download_info();
        let source_url_matches = info.as_ref().map(|info| info.source_url == config.source_url).unwrap_or(false);
        let crc_matches = info.as_ref().map(|info| info.crc32c == config.file_check.expected_crc()).unwrap_or(false);
        if !(source_url_matches && crc_matches) {
            task.cancel();
            continue;
        }
        if live_match.is_none() {
            live_match = Some(task);
        } else {
            task.cancel();
        }
    }
    live_match
}

fn is_live_task_state(state: NSURLSessionTaskState) -> bool {
    matches!(state, NSURLSessionTaskState::Running | NSURLSessionTaskState::Suspended)
}

fn automatic_session_configuration() -> Retained<NSURLSessionConfiguration> {
    let bundle_id = NSBundle::mainBundle().bundleIdentifier().unwrap_or_default().to_string();
    if bundle_id.is_empty() {
        NSURLSessionConfiguration::ephemeralSessionConfiguration()
    } else {
        let session_id = NSString::from_str(&format!("{bundle_id}.trymirai.download-manager"));
        let configuration = NSURLSessionConfiguration::backgroundSessionConfigurationWithIdentifier(&session_id);
        configuration.setSessionSendsLaunchEvents(true);
        configuration.setDiscretionary(false);
        configuration.setWaitsForConnectivity(true);
        configuration
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
        _destination_lease: &DestinationLockLease,
    ) -> Result<AppleActiveTask, AppleBackendError> {
        let ns_url = NSURL::URLWithString(&NSString::from_str(&config.source_url)).ok_or(AppleBackendError::BadUrl)?;
        let task = self.session.downloadTaskWithURL(&ns_url);
        self.prepare_task(&task, Arc::clone(&config), generation, backend_event_sender);
        task.resume();
        Ok(AppleActiveTask::new(task, Arc::clone(&self.event_registry), config.download_id))
    }

    async fn resume(
        &self,
        config: Arc<DownloadConfig>,
        generation: ActiveDownloadGeneration,
        resume_artifact_path: &Path,
        backend_event_sender: BackendEventSender,
        _destination_lease: &DestinationLockLease,
    ) -> Result<AppleActiveTask, AppleBackendError> {
        let resume_data =
            tokio::fs::read(resume_artifact_path).await.map_err(|error| AppleBackendError::Io(error.to_string()))?;
        let ns_data = NSData::with_bytes(&resume_data);
        let task = self.session.downloadTaskWithResumeData(&ns_data);
        self.prepare_task(&task, Arc::clone(&config), generation, backend_event_sender);
        task.resume();
        Ok(AppleActiveTask::new(task, Arc::clone(&self.event_registry), config.download_id))
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
                (config.download_id, task.task_identifier()),
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
