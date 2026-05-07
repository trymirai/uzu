use std::sync::Arc;

use tokio::runtime::Handle as TokioHandle;
use tokio_stream::wrappers::BroadcastStream as TokioBroadcastStream;

use crate::{
    DownloadError, DownloadEvent, FileCheck, FileDownloadManager, FileDownloadTask, LockFileState,
    backends::common::{Backend, DownloadManagerState, Startup},
    check_lock_file, compute_download_id,
    download_log_event::{DownloadLogEvent, record_download_log_event},
    file_download_task::CachedFileDownloadTask,
    file_download_task_actor::GenericFileDownloadTask,
    lock_manager::{DestinationLockLease, lock_path_for_destination},
    reducer::InitialLifecycleState,
};

#[derive(Clone, Debug)]
pub struct DownloadManager<B: Backend> {
    state: DownloadManagerState,
    context: Arc<B::Context>,
}

impl<B: Backend> DownloadManager<B> {
    pub fn from_tokio_handle(tokio_handle: TokioHandle) -> Result<Self, DownloadError> {
        let context = B::create_context(tokio_handle.clone())?;
        let state = DownloadManagerState::new(B::manager_suffix());
        record_download_log_event(DownloadLogEvent::ManagerCreated {
            manager_id: state.manager_id.clone(),
        });
        Ok(Self {
            state,
            context: Arc::new(context),
        })
    }
}

#[async_trait::async_trait]
impl<B: Backend> FileDownloadManager for DownloadManager<B> {
    fn manager_id(&self) -> &str {
        &self.state.manager_id
    }

    fn subscribe_to_all_downloads(&self) -> TokioBroadcastStream<DownloadEvent> {
        self.state.subscribe_to_all_downloads()
    }

    fn global_broadcast_sender(&self) -> crate::SharedDownloadEventSender {
        self.state.global_broadcast_sender()
    }

    async fn get_all_file_tasks(&self) -> Result<Vec<Arc<dyn FileDownloadTask>>, DownloadError> {
        self.state.get_all_file_tasks().await
    }

    async fn remove_file_task(
        &self,
        download_id: crate::DownloadId,
    ) -> Result<(), DownloadError> {
        let construction_lock = self.state.construction_lock(download_id).await;
        let _construction_guard = construction_lock.lock().await;
        let shutdown_result = match self.state.take_task(download_id).await {
            Some(task) => task.managed().shutdown_for_removal().await,
            None => Ok(()),
        };
        self.state.remove_construction_lock_if_unshared(download_id, &construction_lock).await;
        match shutdown_result {
            Ok(()) | Err(DownloadError::TaskStopped | DownloadError::ChannelClosed) => Ok(()),
            Err(error) => Err(error),
        }
    }

    async fn file_download_task(
        &self,
        source_url: &str,
        destination_path: &std::path::Path,
        file_check: FileCheck,
        expected_bytes: Option<u64>,
    ) -> Result<Arc<dyn FileDownloadTask>, DownloadError> {
        let download_id = compute_download_id(destination_path);
        if let Some(cached_task) = self.state.get_task(download_id).await
            && !cached_task.is_stopped()
        {
            let task = cached_task.public();
            ensure_cached_task_matches(&task, source_url, &file_check, expected_bytes)?;
            return Ok(task);
        }

        let construction_lock = self.state.construction_lock(download_id).await;
        let _construction_guard = construction_lock.lock().await;
        let result = async {
            let cached_task_result = match self.state.get_task(download_id).await {
                Some(cached_task) if !cached_task.is_stopped() => {
                    let task = cached_task.public();
                    ensure_cached_task_matches(&task, source_url, &file_check, expected_bytes)?;
                    Some(Ok(task))
                },
                Some(_) => {
                    let _ = self.state.take_task(download_id).await;
                    None
                },
                None => None,
            };
            if let Some(cached_task_result) = cached_task_result {
                cached_task_result
            } else {
                let startup = observe_startup::<B>(
                    download_id,
                    source_url,
                    destination_path,
                    file_check.clone(),
                    expected_bytes,
                    &self.state.manager_id,
                    self.state.instance_id,
                )
                .await?;
                let (startup, startup_lease) = prepare_startup::<B>(
                    startup,
                    self.context.as_ref(),
                    download_id,
                    source_url,
                    destination_path,
                    file_check,
                    expected_bytes,
                    &self.state.manager_id,
                    self.state.instance_id,
                )
                .await?;
                record_download_log_event(DownloadLogEvent::StartupReconciled {
                    download_id,
                });

                let task = Arc::new(
                    GenericFileDownloadTask::<B>::spawn_with_initial_attachment(
                        startup.config,
                        Arc::clone(&self.context),
                        startup.initial_lifecycle_state,
                        startup.initial_projection,
                        startup.initial_progress,
                        startup_lease,
                    )
                    .await?,
                );
                let public_task: Arc<dyn FileDownloadTask> = task.clone();
                let managed_task = task;
                self.state
                    .insert_task(download_id, CachedFileDownloadTask::new(Arc::clone(&public_task), managed_task))
                    .await;
                record_download_log_event(DownloadLogEvent::TaskSpawned {
                    download_id,
                });
                Ok(public_task)
            }
        }
        .await;
        if result.is_err() {
            self.state.remove_construction_lock_if_unshared(download_id, &construction_lock).await;
        }
        result
    }

    async fn destination_foreign_lock(
        &self,
        destination_path: &std::path::Path,
    ) -> Option<String> {
        let lock_path = lock_path_for_destination(destination_path);
        match check_lock_file(&lock_path, &self.state.manager_id, self.state.instance_id, std::process::id()).await {
            LockFileState::OwnedByOtherApp(info) => Some(info.manager_id),
            _ => None,
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn observe_startup<B: Backend>(
    download_id: crate::DownloadId,
    source_url: &str,
    destination_path: &std::path::Path,
    file_check: FileCheck,
    expected_bytes: Option<u64>,
    manager_id: &str,
    manager_instance_id: uuid::Uuid,
) -> Result<Startup, DownloadError> {
    Startup::observe::<B>(
        download_id,
        source_url,
        destination_path,
        file_check,
        expected_bytes,
        manager_id,
        manager_instance_id,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn prepare_startup<B: Backend>(
    startup: Startup,
    context: &B::Context,
    download_id: crate::DownloadId,
    source_url: &str,
    destination_path: &std::path::Path,
    file_check: FileCheck,
    expected_bytes: Option<u64>,
    manager_id: &str,
    manager_instance_id: uuid::Uuid,
) -> Result<(Startup, Option<DestinationLockLease>), DownloadError> {
    if !startup_requires_destination_lease::<B>(&startup, context).await? {
        return Ok((startup, None));
    }

    let lease =
        match DestinationLockLease::acquire_for_destination(destination_path, manager_id, manager_instance_id).await {
            Ok(lease) => lease,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                let startup = observe_startup::<B>(
                    download_id,
                    source_url,
                    destination_path,
                    file_check,
                    expected_bytes,
                    manager_id,
                    manager_instance_id,
                )
                .await?;
                if startup.lock_state.is_conflict() {
                    return Ok((startup, None));
                }
                return Err(DownloadError::from(error));
            },
            Err(error) => return Err(DownloadError::from(error)),
        };
    let startup = match observe_startup::<B>(
        download_id,
        source_url,
        destination_path,
        file_check,
        expected_bytes,
        manager_id,
        manager_instance_id,
    )
    .await
    {
        Ok(startup) => startup,
        Err(error) => {
            let _ = lease.release().await;
            return Err(error);
        },
    };

    if let Err(error) = startup.apply_actions(&lease).await {
        let _ = lease.release().await;
        return Err(error);
    }

    if startup_can_attach_initial_task::<B>(&startup) {
        Ok((startup, Some(lease)))
    } else {
        lease.release().await?;
        Ok((startup, None))
    }
}

async fn startup_requires_destination_lease<B: Backend>(
    startup: &Startup,
    context: &B::Context,
) -> Result<bool, DownloadError> {
    if startup.lock_state.is_conflict() || !startup.action_plan.is_empty() {
        return Ok(!startup.lock_state.is_conflict());
    }

    if !startup_can_attach_initial_task::<B>(startup) {
        return Ok(false);
    }

    B::has_initial_task_to_claim(context, startup.config.as_ref()).await
}

fn startup_can_attach_initial_task<B: Backend>(startup: &Startup) -> bool {
    B::SUPPORTS_INITIAL_TASK_ATTACHMENT
        && !startup.lock_state.is_conflict()
        && !matches!(startup.initial_lifecycle_state, InitialLifecycleState::Downloaded { .. })
}

fn ensure_cached_task_matches(
    cached: &Arc<dyn FileDownloadTask>,
    source_url: &str,
    file_check: &FileCheck,
    expected_bytes: Option<u64>,
) -> Result<(), DownloadError> {
    if cached.source_url() != source_url {
        return Err(DownloadError::ConflictingConfig(format!(
            "{} already requested with source_url {:?}; cannot reuse for {:?}",
            cached.destination().display(),
            cached.source_url(),
            source_url,
        )));
    }
    if cached.file_check() != file_check {
        return Err(DownloadError::ConflictingConfig(format!(
            "{} already requested with a different file_check",
            cached.destination().display(),
        )));
    }
    if cached.expected_bytes() != expected_bytes {
        return Err(DownloadError::ConflictingConfig(format!(
            "{} already requested with expected_bytes {:?}; got {:?}",
            cached.destination().display(),
            cached.expected_bytes(),
            expected_bytes,
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        path::{Path, PathBuf},
        sync::Arc,
    };

    use chrono::Utc;
    use tokio::sync::broadcast::{Sender as TokioBroadcastSender, channel as tokio_broadcast_channel};
    use tokio_stream::wrappers::BroadcastStream as TokioBroadcastStream;
    use uuid::Uuid;

    use super::*;
    use crate::{
        DownloadId, FileDownloadState, LockFileInfo,
        backends::{
            common::InitialTaskAttachment,
            universal::{UniversalBackend, UniversalBackendContext, UniversalBackendError},
        },
        file_download_task::ManagedFileDownloadTask,
        lock_manager::lock_path_for_destination,
        traits::{
            ActiveDownloadGeneration, ActiveTask, BackendContext, BackendEventSender, CancelOutcome, DownloadBackend,
            DownloadConfig,
        },
    };

    #[derive(Debug)]
    struct ShutdownErrorTask {
        download_id: DownloadId,
        source_url: String,
        destination: PathBuf,
        file_check: FileCheck,
        broadcast_sender: TokioBroadcastSender<FileDownloadState>,
    }

    impl ShutdownErrorTask {
        fn new(
            download_id: DownloadId,
            destination: PathBuf,
        ) -> Self {
            let (broadcast_sender, _) = tokio_broadcast_channel(1);
            Self {
                download_id,
                source_url: "http://example.invalid/file".to_string(),
                destination,
                file_check: FileCheck::None,
                broadcast_sender,
            }
        }
    }

    #[derive(Clone, Debug)]
    struct FailingAttachmentBackend;

    #[derive(Debug)]
    struct FailingAttachmentContext;

    #[derive(Debug)]
    struct FailingAttachmentTask;

    impl DownloadBackend for FailingAttachmentBackend {
        type Context = FailingAttachmentContext;
        type ActiveTask = FailingAttachmentTask;
        type Error = UniversalBackendError;
    }

    #[async_trait::async_trait]
    impl BackendContext for FailingAttachmentContext {
        type Backend = FailingAttachmentBackend;

        async fn download(
            &self,
            _config: Arc<DownloadConfig>,
            _generation: ActiveDownloadGeneration,
            _backend_event_sender: BackendEventSender,
            _destination_lease: &DestinationLockLease,
        ) -> Result<FailingAttachmentTask, UniversalBackendError> {
            Ok(FailingAttachmentTask)
        }

        async fn resume(
            &self,
            _config: Arc<DownloadConfig>,
            _generation: ActiveDownloadGeneration,
            _resume_artifact_path: &Path,
            _backend_event_sender: BackendEventSender,
            _destination_lease: &DestinationLockLease,
        ) -> Result<FailingAttachmentTask, UniversalBackendError> {
            Ok(FailingAttachmentTask)
        }
    }

    #[async_trait::async_trait]
    impl ActiveTask for FailingAttachmentTask {
        type Backend = FailingAttachmentBackend;

        async fn pause(
            self,
            _destination: &Path,
        ) -> Result<PathBuf, UniversalBackendError> {
            Err(UniversalBackendError::Io("synthetic pause failure".to_string()))
        }

        async fn cancel(
            self,
            _destination: &Path,
        ) -> CancelOutcome {
            CancelOutcome::BestEffort
        }
    }

    #[async_trait::async_trait]
    impl Backend for FailingAttachmentBackend {
        const RESUME_ARTIFACT_EXTENSION: &'static str = "part";
        const SUPPORTS_INITIAL_TASK_ATTACHMENT: bool = true;

        fn manager_suffix() -> &'static str {
            "failing-attachment"
        }

        fn create_context(_tokio_handle: TokioHandle) -> Result<FailingAttachmentContext, DownloadError> {
            Ok(FailingAttachmentContext)
        }

        async fn initial_task_attachment(
            _context: &FailingAttachmentContext,
            _config: Arc<DownloadConfig>,
            _generation: ActiveDownloadGeneration,
            _backend_event_sender: BackendEventSender,
            _destination_lease: &DestinationLockLease,
        ) -> Result<InitialTaskAttachment<Self>, DownloadError> {
            Err(DownloadError::Backend("synthetic attachment failure".to_string()))
        }

        async fn has_initial_task_to_claim(
            _context: &FailingAttachmentContext,
            _config: &DownloadConfig,
        ) -> Result<bool, DownloadError> {
            Ok(true)
        }
    }

    #[async_trait::async_trait]
    impl FileDownloadTask for ShutdownErrorTask {
        fn download_id(&self) -> DownloadId {
            self.download_id
        }

        fn source_url(&self) -> &str {
            &self.source_url
        }

        fn destination(&self) -> &Path {
            &self.destination
        }

        fn file_check(&self) -> &FileCheck {
            &self.file_check
        }

        fn expected_bytes(&self) -> Option<u64> {
            None
        }

        async fn download(&self) -> Result<(), DownloadError> {
            Ok(())
        }

        async fn pause(&self) -> Result<(), DownloadError> {
            Ok(())
        }

        async fn cancel(&self) -> Result<(), DownloadError> {
            Ok(())
        }

        async fn state(&self) -> FileDownloadState {
            FileDownloadState::not_downloaded(0)
        }

        async fn progress(&self) -> Result<TokioBroadcastStream<FileDownloadState>, DownloadError> {
            Ok(TokioBroadcastStream::new(self.broadcast_sender.subscribe()))
        }

        async fn start_listening(
            &self,
            _global_broadcast: crate::DownloadEventSender,
        ) {
        }

        async fn stop_listening(&self) {}

        async fn wait(&self) {}

        fn broadcast_sender(&self) -> TokioBroadcastSender<FileDownloadState> {
            self.broadcast_sender.clone()
        }
    }

    #[async_trait::async_trait]
    impl ManagedFileDownloadTask for ShutdownErrorTask {
        async fn shutdown_for_removal(&self) -> Result<(), DownloadError> {
            Err(DownloadError::Backend("synthetic shutdown failure".to_string()))
        }

        fn is_stopped(&self) -> bool {
            false
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn prepare_startup_reobserves_lock_before_cleanup_actions() -> Result<(), Box<dyn std::error::Error>> {
        let temporary_directory = tempfile::tempdir()?;
        let destination = temporary_directory.path().join("model.bin");
        let crc_path = std::path::PathBuf::from(format!("{}.crc", destination.display()));
        tokio::fs::write(&destination, b"corrupt").await?;
        tokio::fs::write(&crc_path, b"stale-crc").await?;

        let download_id = compute_download_id(&destination);
        let manager_instance_id = Uuid::new_v4();
        let initial_startup = observe_startup::<UniversalBackend>(
            download_id,
            "http://example.invalid/model.bin",
            &destination,
            FileCheck::CRC("00000000".to_string()),
            Some("corrupt".len() as u64),
            "self-manager",
            manager_instance_id,
        )
        .await?;
        assert!(
            !initial_startup.action_plan.is_empty(),
            "precondition: stale startup observation should want to delete invalid destination state",
        );

        let foreign_owner = LockFileInfo {
            manager_id: "foreign-manager".to_string(),
            instance_id: Uuid::new_v4(),
            acquired_at: Utc::now(),
            process_id: std::process::id(),
        };
        tokio::fs::write(lock_path_for_destination(&destination), serde_json::to_vec(&foreign_owner)?).await?;

        let (reobserved_startup, startup_lease) = prepare_startup::<UniversalBackend>(
            initial_startup,
            &UniversalBackendContext::new(TokioHandle::current()),
            download_id,
            "http://example.invalid/model.bin",
            &destination,
            FileCheck::CRC("00000000".to_string()),
            Some("corrupt".len() as u64),
            "self-manager",
            manager_instance_id,
        )
        .await?;

        assert!(startup_lease.is_none(), "foreign lock must prevent startup from taking a lease");
        assert!(matches!(reobserved_startup.lock_state, LockFileState::OwnedByOtherApp(_)));
        assert!(destination.exists(), "cleanup must not delete a destination after a foreign lock appears");
        assert!(crc_path.exists(), "cleanup must not delete CRC cache after a foreign lock appears");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn remove_file_task_cleans_construction_lock_when_shutdown_fails() -> Result<(), Box<dyn std::error::Error>> {
        let state = DownloadManagerState::new("test");
        let manager = DownloadManager::<UniversalBackend> {
            state: state.clone(),
            context: Arc::new(UniversalBackendContext::new(TokioHandle::current())),
        };
        let download_id = Uuid::new_v4();
        let temporary_directory = tempfile::tempdir()?;
        let destination = temporary_directory.path().join("model.bin");
        let construction_lock = state.construction_lock(download_id).await;
        let weak_construction_lock = Arc::downgrade(&construction_lock);
        drop(construction_lock);
        let task = Arc::new(ShutdownErrorTask::new(download_id, destination));
        let public_task: Arc<dyn FileDownloadTask> = task.clone();
        let managed_task = task;
        state.insert_task(download_id, CachedFileDownloadTask::new(public_task, managed_task)).await;

        let result = manager.remove_file_task(download_id).await;

        assert!(
            matches!(result, Err(DownloadError::Backend(message)) if message == "synthetic shutdown failure"),
            "shutdown failures should still be reported to the caller",
        );
        assert!(
            weak_construction_lock.upgrade().is_none(),
            "construction lock entry must be cleaned even when task shutdown fails",
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cleans_lock_after_attachment_error() -> Result<(), Box<dyn std::error::Error>> {
        let state = DownloadManagerState::new("test");
        let manager = DownloadManager::<FailingAttachmentBackend> {
            state: state.clone(),
            context: Arc::new(FailingAttachmentContext),
        };
        let temporary_directory = tempfile::tempdir()?;
        let destination = temporary_directory.path().join("model.bin");
        let download_id = compute_download_id(&destination);
        let lock_path = lock_path_for_destination(&destination);
        let construction_lock = state.construction_lock(download_id).await;
        let weak_construction_lock = Arc::downgrade(&construction_lock);
        drop(construction_lock);

        let result = manager
            .file_download_task("http://example.invalid/model.bin", &destination, FileCheck::None, Some(100))
            .await;

        assert!(
            matches!(result, Err(DownloadError::Backend(message)) if message == "synthetic attachment failure"),
            "initial attachment failure should be returned to the caller",
        );
        assert!(
            weak_construction_lock.upgrade().is_none(),
            "construction lock entry must be cleaned when task construction fails",
        );
        assert!(!lock_path.exists(), "startup lease must be released when initial attachment fails");
        Ok(())
    }
}
