use std::sync::Arc;

use kiban::rt::RuntimeHandle;
use tokio_stream::wrappers::BroadcastStream as TokioBroadcastStream;

use crate::{
    DownloadError, DownloadEvent, FileCheck, FileDownloadManager, FileDownloadTask, LockFileState,
    backends::common::{Backend, DownloadManagerState, Startup},
    check_lock_file, compute_download_id,
    download_log_event::{DownloadLogEvent, log},
    file_download_task::CachedFileDownloadTask,
    file_download_task_actor::GenericFileDownloadTask,
    lock_manager::{DestinationLockLease, lock_path_for_destination},
    reducer::InitialLifecycleState,
};

#[derive(Clone, Debug)]
pub struct DownloadManager<B: Backend> {
    pub(crate) state: DownloadManagerState,
    pub(crate) context: Arc<B::Context>,
}

impl<B: Backend> DownloadManager<B> {
    pub fn from_runtime_handle(runtime_handle: RuntimeHandle) -> Result<Self, DownloadError> {
        let context = B::create_context(runtime_handle)?;
        let state = DownloadManagerState::new(B::manager_suffix());
        log(DownloadLogEvent::ManagerCreated {
            manager_id: state.manager_id.clone(),
        });
        Ok(Self {
            state,
            context: Arc::new(context),
        })
    }
}

#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
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

                log(DownloadLogEvent::StartupReconciled {
                    download_id,
                    initial_lifecycle_state: startup.initial_lifecycle_state.name(),
                    action_count: startup.action_plan.as_slice().len(),
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
                log(DownloadLogEvent::TaskSpawned {
                    download_id,
                });
                let public_task: Arc<dyn FileDownloadTask> = task.clone();
                let managed_task = task;
                self.state
                    .insert_task(download_id, CachedFileDownloadTask::new(Arc::clone(&public_task), managed_task))
                    .await;
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
        match check_lock_file(&lock_path, &self.state.manager_id, self.state.instance_id, kiban::process::id()).await {
            LockFileState::OwnedByOtherApp(info) => Some(info.manager_id),
            _ => None,
        }
    }
}

pub(crate) async fn observe_startup<B: Backend>(
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

pub(crate) async fn prepare_startup<B: Backend>(
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
        && !matches!(startup.initial_lifecycle_state, InitialLifecycleState::Downloaded)
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
