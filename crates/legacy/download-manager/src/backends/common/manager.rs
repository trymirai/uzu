use std::sync::Arc;

use kiban::{fs, rt::RuntimeHandle};
use tokio_stream::wrappers::BroadcastStream as TokioBroadcastStream;

use crate::{
    DownloadError, DownloadEvent, FileCheck, FileDownloadManager, FileDownloadTask, HttpDownloadRequest, LockFileState,
    backends::common::{Backend, DownloadManagerState, Startup},
    compute_download_id,
    download_log_event::{DownloadLogEvent, log},
    file_download_task::{CachedFileDownloadTask, InactiveTaskShutdown},
    file_download_task_actor::GenericFileDownloadTask,
    lock_manager::{DestinationLockLease, check_lock_file, lock_path_for_destination},
    reducer::InitialLifecycleState,
    traits::DownloadConfig,
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

    async fn http_file_download_task(
        &self,
        request: HttpDownloadRequest,
        destination_path: &std::path::Path,
        file_check: FileCheck,
        expected_bytes: Option<u64>,
    ) -> Result<Arc<dyn FileDownloadTask>, DownloadError> {
        let download_id = compute_download_id(destination_path);
        let artifact_root = DownloadConfig::default_artifact_root(destination_path, download_id);
        self.http_file_download_task_with_artifact_root(
            request,
            destination_path,
            file_check,
            expected_bytes,
            &artifact_root,
        )
        .await
    }

    async fn http_file_download_task_with_artifact_root(
        &self,
        request: HttpDownloadRequest,
        destination_path: &std::path::Path,
        file_check: FileCheck,
        expected_bytes: Option<u64>,
        artifact_root: &std::path::Path,
    ) -> Result<Arc<dyn FileDownloadTask>, DownloadError> {
        request.validate()?;
        let download_id = compute_download_id(destination_path);
        if let Some(cached_task) = self.state.get_task(download_id).await
            && !cached_task.is_stopped()
        {
            let task = cached_task.public();
            if cached_task.artifact_root() == artifact_root
                && cached_task_config_conflict(&task, &request, &file_check, expected_bytes).is_none()
            {
                return Ok(task);
            }
        }

        let construction_lock = self.state.construction_lock(download_id).await;
        let _construction_guard = construction_lock.lock().await;
        let result = async {
            let cached_task_result = match self.state.get_task(download_id).await {
                Some(cached_task) if !cached_task.is_stopped() => {
                    let task = cached_task.public();
                    let conflict = if cached_task.artifact_root() == artifact_root {
                        cached_task_config_conflict(&task, &request, &file_check, expected_bytes)
                    } else {
                        Some(DownloadError::ConflictingConfig(format!(
                            "{} already uses a different manager artifact root",
                            task.destination().display(),
                        )))
                    };
                    match conflict {
                        None => Some(Ok(task)),
                        Some(conflict) => match cached_task.managed().shutdown_for_replacement_if_inactive().await? {
                            InactiveTaskShutdown::Stopped => {
                                let _ = self.state.take_task(download_id).await;
                                None
                            },
                            InactiveTaskShutdown::Active => Some(Err(conflict)),
                        },
                    }
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
                    request.clone(),
                    destination_path,
                    artifact_root,
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
                    &request,
                    destination_path,
                    artifact_root,
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
                    .insert_task(
                        download_id,
                        CachedFileDownloadTask::new(
                            Arc::clone(&public_task),
                            managed_task,
                            artifact_root.to_path_buf(),
                        ),
                    )
                    .await;
                Ok(public_task)
            }
        }
        .await;
        self.state.remove_construction_lock_if_unshared(download_id, &construction_lock).await;
        result
    }

    async fn open_existing_http_file_download_task_with_artifact_root(
        &self,
        request: HttpDownloadRequest,
        destination_path: &std::path::Path,
        file_check: FileCheck,
        expected_bytes: Option<u64>,
        artifact_root: &std::path::Path,
    ) -> Result<Option<Arc<dyn FileDownloadTask>>, DownloadError> {
        enum OpenExistingDecision {
            Return(Option<Arc<dyn FileDownloadTask>>),
            Materialize,
        }

        request.validate()?;
        let download_id = compute_download_id(destination_path);
        let construction_lock = self.state.construction_lock(download_id).await;
        let construction_guard = construction_lock.lock().await;
        let decision = async {
            if let Some(cached_task) = self.state.get_task(download_id).await {
                if cached_task.is_stopped() {
                    let _ = self.state.take_task(download_id).await;
                } else {
                    let task = cached_task.public();
                    let conflict = if cached_task.artifact_root() == artifact_root {
                        cached_task_config_conflict(&task, &request, &file_check, expected_bytes)
                    } else {
                        Some(DownloadError::ConflictingConfig(format!(
                            "{} already uses a different manager artifact root",
                            task.destination().display(),
                        )))
                    };
                    let shutdown = match conflict {
                        None => cached_task.managed().shutdown_preserving_artifacts_if_inactive().await?,
                        Some(conflict) => match cached_task.managed().shutdown_for_replacement_if_inactive().await? {
                            InactiveTaskShutdown::Active => {
                                return Err(conflict);
                            },
                            InactiveTaskShutdown::Stopped => InactiveTaskShutdown::Stopped,
                        },
                    };
                    match shutdown {
                        InactiveTaskShutdown::Active => return Ok(OpenExistingDecision::Return(Some(task))),
                        InactiveTaskShutdown::Stopped => {
                            let _ = self.state.take_task(download_id).await;
                        },
                    }
                }
            }

            let config = DownloadConfig {
                download_id,
                request: request.clone(),
                destination: destination_path.to_path_buf(),
                artifact_root: artifact_root.to_path_buf(),
                file_check: file_check.clone(),
                expected_bytes,
                manager_id: self.state.manager_id.clone(),
                manager_instance_id: self.state.instance_id,
            };
            let has_local_state = fs::asyn::try_exists(destination_path).await?
                || fs::asyn::try_exists(artifact_root).await?
                || fs::asyn::try_exists(config.lock_path()).await?;
            let has_background_task = B::SUPPORTS_INITIAL_TASK_ATTACHMENT
                && B::has_initial_task_to_claim(self.context.as_ref(), &config).await?;
            if has_local_state || has_background_task {
                Ok(OpenExistingDecision::Materialize)
            } else {
                Ok(OpenExistingDecision::Return(None))
            }
        }
        .await;

        drop(construction_guard);
        let result = match decision {
            Ok(OpenExistingDecision::Return(task)) => Ok(task),
            Ok(OpenExistingDecision::Materialize) => {
                // Re-enter through the regular constructor after releasing the
                // per-file lock. Concurrent callers still serialize on the same
                // cached lock.
                self.http_file_download_task_with_artifact_root(
                    request,
                    destination_path,
                    file_check,
                    expected_bytes,
                    artifact_root,
                )
                .await
                .map(Some)
            },
            Err(error) => Err(error),
        };
        self.state.remove_construction_lock_if_unshared(download_id, &construction_lock).await;
        result
    }

    async fn release_file_task_if_inactive(
        &self,
        task: Arc<dyn FileDownloadTask>,
    ) -> Result<(), DownloadError> {
        let download_id = task.download_id();
        let construction_lock = self.state.construction_lock(download_id).await;
        let _construction_guard = construction_lock.lock().await;
        let result = async {
            let Some(cached_task) = self.state.get_task(download_id).await else {
                return Ok(());
            };
            if !Arc::ptr_eq(&cached_task.public(), &task) {
                return Ok(());
            }

            if cached_task.is_stopped()
                || cached_task.managed().shutdown_preserving_artifacts_if_inactive().await?
                    == InactiveTaskShutdown::Stopped
            {
                let _ = self.state.take_task(download_id).await;
            }
            Ok(())
        }
        .await;
        self.state.remove_construction_lock_if_unshared(download_id, &construction_lock).await;
        result
    }
}

pub(crate) async fn observe_startup<B: Backend>(
    download_id: crate::DownloadId,
    request: HttpDownloadRequest,
    destination_path: &std::path::Path,
    artifact_root: &std::path::Path,
    file_check: FileCheck,
    expected_bytes: Option<u64>,
    manager_id: &str,
    manager_instance_id: uuid::Uuid,
) -> Result<Startup, DownloadError> {
    Startup::observe::<B>(
        download_id,
        request,
        destination_path,
        artifact_root,
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
    request: &HttpDownloadRequest,
    destination_path: &std::path::Path,
    artifact_root: &std::path::Path,
    file_check: FileCheck,
    expected_bytes: Option<u64>,
    manager_id: &str,
    manager_instance_id: uuid::Uuid,
) -> Result<(Startup, Option<DestinationLockLease>), DownloadError> {
    if !startup_requires_destination_lease::<B>(&startup, context).await? {
        return Ok((startup, None));
    }

    let lock_path = startup.config.lock_path();
    let lease = match DestinationLockLease::acquire(&lock_path, manager_id, manager_instance_id).await {
        Ok(lease) => lease,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let startup = observe_startup::<B>(
                download_id,
                request.clone(),
                destination_path,
                artifact_root,
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
        request.clone(),
        destination_path,
        artifact_root,
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

fn cached_task_config_conflict(
    cached: &Arc<dyn FileDownloadTask>,
    request: &HttpDownloadRequest,
    file_check: &FileCheck,
    expected_bytes: Option<u64>,
) -> Option<DownloadError> {
    if cached.http_request() != *request {
        return Some(DownloadError::ConflictingConfig(format!(
            "{} already requested with a different HTTP request",
            cached.destination().display(),
        )));
    }
    if cached.file_check() != file_check {
        return Some(DownloadError::ConflictingConfig(format!(
            "{} already requested with a different file_check",
            cached.destination().display(),
        )));
    }
    if cached.expected_bytes() != expected_bytes {
        return Some(DownloadError::ConflictingConfig(format!(
            "{} already requested with expected_bytes {:?}; got {:?}",
            cached.destination().display(),
            cached.expected_bytes(),
            expected_bytes,
        )));
    }
    None
}
