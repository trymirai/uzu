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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use mock_registry::{Behavior, MockRegistry};
    use tokio::{
        runtime::Handle as TokioHandle,
        sync::{
            Mutex as TokioMutex,
            mpsc::{Receiver as TokioMpscReceiver, channel as tokio_mpsc_channel},
            watch::channel as tokio_watch_channel,
        },
    };
    use uuid::Uuid;

    use crate::{
        DownloadId, FileCheck,
        backends::apple::AppleBackendContext,
        compute_download_id,
        file_download_task_actor::{BackendEvent, PendingProgressSlot},
        lock_manager::DestinationLockLease,
        traits::{ActiveDownloadGeneration, BackendContext, BackendEventSender, DownloadConfig},
    };

    fn backend_event_sender() -> (BackendEventSender, TokioMpscReceiver<BackendEvent>) {
        let (backend_event_sender, backend_event_receiver) = tokio_mpsc_channel(64);
        let pending_progress = Arc::new(TokioMutex::new(PendingProgressSlot::default()));
        let (progress_waker_sender, _progress_waker_receiver) = tokio_watch_channel(());
        (BackendEventSender::new(backend_event_sender, pending_progress, progress_waker_sender), backend_event_receiver)
    }

    fn event_sink_count(
        context: &AppleBackendContext,
        download_id: DownloadId,
    ) -> usize {
        context
            .event_registry
            .lock()
            .map(|registry| registry.keys().filter(|(id, _)| *id == download_id).count())
            .unwrap_or(0)
    }

    fn event_sink_task_identifiers(
        context: &AppleBackendContext,
        download_id: DownloadId,
    ) -> Vec<u64> {
        context
            .event_registry
            .lock()
            .map(|registry| {
                registry
                    .keys()
                    .filter_map(|(id, task_identifier)| (*id == download_id).then_some(*task_identifier))
                    .collect()
            })
            .unwrap_or_default()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn dropping_apple_active_task_unregisters_event_sink() -> Result<(), Box<dyn std::error::Error>> {
        let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
        let served_file = registry.file("config.json")?;
        let temporary_directory = tempfile::tempdir()?;
        let destination = temporary_directory.path().join(&served_file.file.name);

        let context = AppleBackendContext::new(TokioHandle::current());
        let download_id = compute_download_id(&destination);
        let config = Arc::new(DownloadConfig {
            download_id,
            source_url: served_file.file.url.clone(),
            destination,
            file_check: FileCheck::None,
            expected_bytes: Some(served_file.file.size as u64),
            manager_id: "test-manager".to_string(),
            manager_instance_id: Uuid::new_v4(),
        });
        let destination_lease = DestinationLockLease::acquire_for_destination(
            &config.destination,
            &config.manager_id,
            config.manager_instance_id,
        )
        .await?;

        let (backend_event_sender, _backend_event_receiver) = backend_event_sender();
        let active_task = context
            .download(Arc::clone(&config), ActiveDownloadGeneration::new(0), backend_event_sender, &destination_lease)
            .await?;
        assert_eq!(
            event_sink_count(&context, download_id),
            1,
            "precondition: exactly one event sink should be registered for download_id after starting download",
        );

        drop(active_task);
        destination_lease.release().await?;

        assert_eq!(
            event_sink_count(&context, download_id),
            0,
            "dropping AppleActiveTask must unregister its event sink",
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn registry_distinguishes_generations_for_same_download_id() -> Result<(), Box<dyn std::error::Error>> {
        let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
        let served_file = registry.file("config.json")?;
        let temporary_directory = tempfile::tempdir()?;
        let destination = temporary_directory.path().join(&served_file.file.name);
        let download_id = compute_download_id(&destination);

        let context = AppleBackendContext::new(TokioHandle::current());
        let config = Arc::new(DownloadConfig {
            download_id,
            source_url: served_file.file.url.clone(),
            destination,
            file_check: FileCheck::None,
            expected_bytes: Some(served_file.file.size as u64),
            manager_id: "test-manager".to_string(),
            manager_instance_id: Uuid::new_v4(),
        });
        let destination_lease = DestinationLockLease::acquire_for_destination(
            &config.destination,
            &config.manager_id,
            config.manager_instance_id,
        )
        .await?;

        let (backend_event_sender_first, _backend_event_receiver_first) = backend_event_sender();
        let first_task = context
            .download(
                Arc::clone(&config),
                ActiveDownloadGeneration::new(0),
                backend_event_sender_first,
                &destination_lease,
            )
            .await?;
        assert_eq!(
            event_sink_task_identifiers(&context, download_id).len(),
            1,
            "first generation must register a sink"
        );

        let (backend_event_sender_second, _backend_event_receiver_second) = backend_event_sender();
        let second_task = context
            .download(
                Arc::clone(&config),
                ActiveDownloadGeneration::new(1),
                backend_event_sender_second,
                &destination_lease,
            )
            .await?;

        let both_keys = event_sink_task_identifiers(&context, download_id);
        assert_eq!(both_keys.len(), 2, "second generation must not overwrite the first sink; got keys: {both_keys:?}",);
        assert_ne!(both_keys[0], both_keys[1], "the two generations must have distinct task identifiers");

        drop(first_task);
        drop(second_task);
        destination_lease.release().await?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn claim_cancels_mismatched_task() -> Result<(), Box<dyn std::error::Error>> {
        let registry = MockRegistry::start_with(Behavior::THROTTLED).await?;
        let served_file = registry.file("tokenizer.json")?;
        let temporary_directory = tempfile::tempdir()?;
        let destination = temporary_directory.path().join(&served_file.file.name);
        let download_id = compute_download_id(&destination);

        let context = AppleBackendContext::new(TokioHandle::current());
        let original_config = Arc::new(DownloadConfig {
            download_id,
            source_url: served_file.file.url.clone(),
            destination: destination.clone(),
            file_check: FileCheck::None,
            expected_bytes: Some(served_file.file.size as u64),
            manager_id: "test-manager".to_string(),
            manager_instance_id: Uuid::new_v4(),
        });
        let destination_lease = DestinationLockLease::acquire_for_destination(
            &original_config.destination,
            &original_config.manager_id,
            original_config.manager_instance_id,
        )
        .await?;

        let (backend_event_sender, _backend_event_receiver) = backend_event_sender();
        let original_task = context
            .download(
                Arc::clone(&original_config),
                ActiveDownloadGeneration::new(0),
                backend_event_sender,
                &destination_lease,
            )
            .await?;

        assert!(
            context.claim_matching_download_task(&original_config).await?.is_some(),
            "precondition: original task should be visible before the mismatched claim",
        );

        let mismatched_config = DownloadConfig {
            source_url: "http://example.invalid/different-url".to_string(),
            ..(*original_config).clone()
        };
        assert!(
            context.has_download_task_to_claim(&mismatched_config).await?,
            "manager startup must take the claim path for a live same-destination task with mismatched metadata",
        );
        assert!(context.claim_matching_download_task(&mismatched_config).await?.is_none());

        let mut cancelled = false;
        for _attempt in 0..50 {
            if context.claim_matching_download_task(&original_config).await?.is_none() {
                cancelled = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        assert!(
            cancelled,
            "claiming a same-destination task with mismatched metadata must cancel the stale URLSession task",
        );
        drop(original_task);
        destination_lease.release().await?;
        Ok(())
    }
}
