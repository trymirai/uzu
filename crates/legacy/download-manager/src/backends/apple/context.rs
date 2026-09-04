use std::{
    collections::HashMap,
    fmt::{Debug, Formatter, Result as FmtResult},
    path::Path,
    sync::{Arc, Mutex},
};

use http::header::{ACCEPT_ENCODING, AUTHORIZATION};
use kiban::rt::RuntimeHandle;
use objc2::rc::Retained;
use objc2_foundation::{
    NSBundle, NSData, NSMutableURLRequest, NSString, NSURL, NSURLSession, NSURLSessionConfiguration,
    NSURLSessionDownloadTask, NSURLSessionTaskState,
};
use tokio::{fs::read as tokio_read, sync::oneshot::channel as tokio_oneshot_channel};

#[cfg(test)]
use crate::DownloadId;
use crate::{
    DownloadInfo, HttpDownloadRequest,
    backends::apple::{
        AppleActiveTask, AppleBackend, AppleBackendError, AppleEventRegistry, AppleEventSink, AppleGetTasksHandler,
        AppleSessionDelegate, task_ext::AppleDownloadTaskExt,
    },
    lock_manager::DestinationLockLease,
    traits::{ActiveDownloadGeneration, BackendContext, BackendEventSender, DownloadConfig},
};

pub struct AppleBackendContext {
    session: Retained<NSURLSession>,
    event_registry: AppleEventRegistry,
    runtime_handle: RuntimeHandle,
}

impl Debug for AppleBackendContext {
    fn fmt(
        &self,
        formatter: &mut Formatter<'_>,
    ) -> FmtResult {
        formatter.debug_struct("AppleBackendContext").finish_non_exhaustive()
    }
}

impl AppleBackendContext {
    pub fn new(runtime_handle: RuntimeHandle) -> Self {
        let event_registry = Arc::new(Mutex::new(HashMap::new()));
        let delegate = AppleSessionDelegate::new(Arc::clone(&event_registry));
        let delegate = AppleSessionDelegate::protocol_object(delegate);
        let session = unsafe {
            NSURLSession::sessionWithConfiguration_delegate_delegateQueue(
                &automatic_session_configuration(),
                Some(&delegate),
                None,
            )
        };

        Self {
            session,
            event_registry,
            runtime_handle,
        }
    }

    pub(crate) async fn claim_matching_download_task(
        &self,
        config: &DownloadConfig,
    ) -> Result<Option<Retained<NSURLSessionDownloadTask>>, AppleBackendError> {
        select_matching_download_task(self.download_tasks().await?, config)
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
    pub(crate) fn event_sink_task_identifiers_for_download(
        &self,
        download_id: DownloadId,
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
) -> Result<Option<Retained<NSURLSessionDownloadTask>>, AppleBackendError> {
    let authorization = authorization_header(&config.request)?;
    let authorization_field = NSString::from_str(AUTHORIZATION.as_str());
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
        let source_url_matches = info.as_ref().map(|info| info.source_url == config.request.url).unwrap_or(false);
        let file_check_matches = info.as_ref().is_some_and(|info| info.resolved_file_check() == config.file_check);
        let authorization_matches = task
            .originalRequest()
            .and_then(|request| request.valueForHTTPHeaderField(&authorization_field))
            .map(|value| value.to_string())
            == authorization;
        if !(source_url_matches && file_check_matches && authorization_matches) {
            task.cancel();
            continue;
        }
        if live_match.is_none() {
            live_match = Some(task);
        } else {
            task.cancel();
        }
    }
    Ok(live_match)
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
        let request = apple_request(&config)?;
        let task = self.session.downloadTaskWithRequest(&request);
        self.prepare_task(&task, Arc::clone(&config), generation, backend_event_sender);
        task.resume();
        Ok(AppleActiveTask::new(
            task,
            Arc::clone(&self.event_registry),
            config.download_id,
            config.request.is_authenticated(),
        ))
    }

    async fn resume(
        &self,
        config: Arc<DownloadConfig>,
        generation: ActiveDownloadGeneration,
        resume_artifact_path: &Path,
        backend_event_sender: BackendEventSender,
        destination_lease: &DestinationLockLease,
    ) -> Result<AppleActiveTask, AppleBackendError> {
        if config.request.is_authenticated() {
            return self.download(config, generation, backend_event_sender, destination_lease).await;
        }
        let resume_data =
            tokio_read(resume_artifact_path).await.map_err(|error| AppleBackendError::Io(error.to_string()))?;
        let task = if resume_data.is_empty() {
            let request = apple_request(&config)?;
            self.session.downloadTaskWithRequest(&request)
        } else {
            let ns_data = NSData::with_bytes(&resume_data);
            self.session.downloadTaskWithResumeData(&ns_data)
        };
        self.prepare_task(&task, Arc::clone(&config), generation, backend_event_sender);
        task.resume();
        Ok(AppleActiveTask::new(task, Arc::clone(&self.event_registry), config.download_id, false))
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
        let download_info = DownloadInfo::new(
            config.request.url.clone(),
            config.destination.to_string_lossy().to_string(),
            config.file_check.clone(),
        );
        task.set_download_info(&download_info);
        if let Ok(mut registry) = self.event_registry.lock() {
            registry.insert(
                (config.download_id, task.task_identifier()),
                AppleEventSink {
                    generation,
                    destination: config.destination.clone(),
                    expected_bytes: config.expected_bytes,
                    backend_event_sender,
                    runtime_handle: self.runtime_handle.clone(),
                },
            );
        }
    }
}

fn apple_request(config: &DownloadConfig) -> Result<Retained<NSMutableURLRequest>, AppleBackendError> {
    let url = NSURL::URLWithString(&NSString::from_str(&config.request.url)).ok_or(AppleBackendError::BadUrl)?;
    let request = NSMutableURLRequest::requestWithURL(&url);
    if let Some(authorization) = authorization_header(&config.request)? {
        request.setValue_forHTTPHeaderField(
            Some(&NSString::from_str(&authorization)),
            &NSString::from_str(AUTHORIZATION.as_str()),
        );
    }
    request.setValue_forHTTPHeaderField(
        Some(&NSString::from_str("identity")),
        &NSString::from_str(ACCEPT_ENCODING.as_str()),
    );
    Ok(request)
}

fn authorization_header(request: &HttpDownloadRequest) -> Result<Option<String>, AppleBackendError> {
    request
        .bearer_token()
        .map(|token| token.map(|token| format!("Bearer {token}")))
        .map_err(|error| AppleBackendError::Authentication(error.to_string()))
}
