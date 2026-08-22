use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, Mutex},
};

use kiban::rt::RuntimeHandle;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{
    NSBundle, NSData, NSMutableURLRequest, NSString, NSURL, NSURLSession, NSURLSessionConfiguration,
    NSURLSessionDelegate, NSURLSessionDownloadTask, NSURLSessionTaskState,
};
use tokio::sync::oneshot::channel as tokio_oneshot_channel;

use super::active_task::write_resume_data;
use crate::{
    backends::{
        apple::{
            AppleActiveTask, AppleBackend, AppleBackendError, AppleEventRegistry, AppleEventSink, AppleGetTasksHandler,
            AppleSessionDelegate, AppleSinkKey, DestinationInstallBarrier, task_ext::AppleDownloadTaskExt,
        },
        common::ensure_owned_directory,
    },
    lock_manager::DestinationLockLease,
    recovery_metadata::{RecoveryMetadata, prepare_fresh_recovery, prepare_resume_recovery},
    traits::{ActiveDownloadGeneration, BackendContext, BackendEventSender, DownloadConfig},
};

pub struct AppleBackendContext {
    session: Retained<NSURLSession>,
    authenticated_session: Retained<NSURLSession>,
    _delegate: Retained<AppleSessionDelegate>,
    _delegate_protocol_object: Retained<ProtocolObject<dyn NSURLSessionDelegate>>,
    event_registry: AppleEventRegistry,
    runtime_handle: RuntimeHandle,
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
    pub fn new(runtime_handle: RuntimeHandle) -> Self {
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
        let authenticated_session = unsafe {
            NSURLSession::sessionWithConfiguration_delegate_delegateQueue(
                &NSURLSessionConfiguration::ephemeralSessionConfiguration(),
                Some(&delegate_protocol_object),
                None,
            )
        };

        Self {
            session,
            authenticated_session,
            _delegate: delegate,
            _delegate_protocol_object: delegate_protocol_object,
            event_registry,
            runtime_handle,
        }
    }

    pub(crate) async fn claim_matching_download_task(
        &self,
        config: &DownloadConfig,
    ) -> Result<Option<Retained<NSURLSessionDownloadTask>>, AppleBackendError> {
        if config.request.headers.has_authorization() {
            self.cancel_legacy_background_tasks(config).await?;
            return Ok(None);
        }
        self.find_download_task(config).await
    }

    pub(crate) async fn has_download_task_to_claim(
        &self,
        config: &DownloadConfig,
    ) -> Result<bool, AppleBackendError> {
        if config.request.headers.has_authorization() {
            self.cancel_legacy_background_tasks(config).await?;
            return Ok(false);
        }
        let download_tasks = self.download_tasks().await?;
        Ok(download_tasks
            .iter()
            .any(|task| task.download_id() == Some(config.download_id) && is_live_task_state(task.state())))
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
    ) -> DestinationInstallBarrier {
        self.prepare_task(&self.session, task, config, generation, backend_event_sender)
    }

    pub(crate) fn event_registry(&self) -> AppleEventRegistry {
        Arc::clone(&self.event_registry)
    }

    pub(crate) fn background_session(&self) -> Retained<NSURLSession> {
        self.session.clone()
    }

    async fn cancel_legacy_background_tasks(
        &self,
        config: &DownloadConfig,
    ) -> Result<(), AppleBackendError> {
        for task in self.download_tasks().await? {
            if task.download_id() == Some(config.download_id) && is_live_task_state(task.state()) {
                task.cancel();
            }
        }
        Ok(())
    }

    fn session_for_request(
        &self,
        config: &DownloadConfig,
    ) -> Retained<NSURLSession> {
        if config.request.headers.has_authorization() {
            self.authenticated_session.clone()
        } else {
            self.session.clone()
        }
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
        let request_matches = task.recovery_metadata().is_some_and(|metadata| {
            metadata.matches_request(&config.request.url, config.expected_bytes, &config.file_check)
        });
        if !request_matches {
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
        destination_lease: &DestinationLockLease,
    ) -> Result<AppleActiveTask, AppleBackendError> {
        ensure_owned_directory(&config.artifact_root).await?;
        let resume_artifact_path = config.resume_artifact_path("resume_data");
        prepare_fresh_recovery(&config, &resume_artifact_path, destination_lease).await?;
        let request = apple_request(config.as_ref())?;
        let session = self.session_for_request(&config);
        let task = session.downloadTaskWithRequest(&request);
        let destination_install_barrier =
            self.prepare_task(&session, &task, Arc::clone(&config), generation, backend_event_sender);
        task.resume();
        Ok(AppleActiveTask::new_for_request(
            task,
            session,
            Arc::clone(&self.event_registry),
            config.download_id,
            resume_artifact_path,
            &config.request.headers,
            destination_install_barrier,
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
        ensure_owned_directory(&config.artifact_root).await?;
        let can_resume = prepare_resume_recovery(&config, resume_artifact_path, destination_lease).await?;
        let mut resume_data = if can_resume {
            tokio::fs::read(resume_artifact_path).await.map_err(|error| AppleBackendError::Io(error.to_string()))?
        } else {
            Vec::new()
        };
        let session = self.session_for_request(&config);
        if config.request.headers.has_authorization() {
            resume_data.clear();
            write_resume_data(resume_artifact_path, &[]).await?;
        }
        let task = if resume_data.is_empty() {
            let request = apple_request(config.as_ref())?;
            session.downloadTaskWithRequest(&request)
        } else {
            let ns_data = NSData::with_bytes(&resume_data);
            session.downloadTaskWithResumeData(&ns_data)
        };
        let destination_install_barrier =
            self.prepare_task(&session, &task, Arc::clone(&config), generation, backend_event_sender);
        task.resume();
        Ok(AppleActiveTask::new_for_request(
            task,
            session,
            Arc::clone(&self.event_registry),
            config.download_id,
            config.resume_artifact_path("resume_data"),
            &config.request.headers,
            destination_install_barrier,
        ))
    }
}

impl AppleBackendContext {
    fn prepare_task(
        &self,
        session: &NSURLSession,
        task: &NSURLSessionDownloadTask,
        config: Arc<DownloadConfig>,
        generation: ActiveDownloadGeneration,
        backend_event_sender: BackendEventSender,
    ) -> DestinationInstallBarrier {
        let recovery_metadata = persisted_recovery_metadata(config.as_ref());
        task.set_recovery_metadata(&recovery_metadata);
        let destination_install_barrier = DestinationInstallBarrier::default();
        if let Ok(mut registry) = self.event_registry.lock() {
            registry.insert(
                AppleSinkKey::new(session, config.download_id, task.task_identifier()),
                AppleEventSink {
                    generation,
                    destination: config.destination.clone(),
                    installation_artifact: config.installation_artifact_path(),
                    backend_event_sender,
                    runtime_handle: self.runtime_handle.clone(),
                    destination_install_barrier: destination_install_barrier.clone(),
                },
            );
        }
        destination_install_barrier
    }
}

fn apple_request(config: &DownloadConfig) -> Result<Retained<NSMutableURLRequest>, AppleBackendError> {
    let url = NSURL::URLWithString(&NSString::from_str(&config.request.url)).ok_or(AppleBackendError::BadUrl)?;
    let request = NSMutableURLRequest::requestWithURL(&url);
    for (name, value) in config.request.headers.as_header_map() {
        let value = value.to_str().map_err(|_| AppleBackendError::InvalidRequestHeader)?;
        request.setValue_forHTTPHeaderField(Some(&NSString::from_str(value)), &NSString::from_str(name.as_str()));
    }
    Ok(request)
}

fn persisted_recovery_metadata(config: &DownloadConfig) -> RecoveryMetadata {
    RecoveryMetadata::new(
        config.download_id,
        &config.request.url,
        &config.destination,
        config.expected_bytes,
        config.file_check.clone(),
    )
}

#[cfg(test)]
#[path = "../../../tests/unit/backends/apple/context_test.rs"]
mod tests;
