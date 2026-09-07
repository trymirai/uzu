use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use kiban::rt::RuntimeHandle;
use tokio::sync::{Mutex as TokioMutex, watch::Receiver as TokioWatchReceiver};
use uuid::Uuid;

use crate::{
    DownloadError, DownloadId, DownloadManager, DownloadState, DownloadTask, DownloadTaskKind, DownloadTaskRequest,
    GroupDownloadTask, LockFileState,
    backends::common::{Backend, CachedDownloadTask, DownloadConfig, Startup},
    download_log_event::{DownloadLogEvent, log},
    file_download_task_actor::spawn_file_download_task,
    lock_manager::DestinationLockLease,
    reducer::InitialLifecycleState,
};

pub struct BackendDownloadManager<B: Backend> {
    manager_id: String,
    instance_id: Uuid,
    context: Arc<B::Context>,
    tasks: TokioMutex<HashMap<DownloadId, CachedDownloadTask>>,
    construction_locks: TokioMutex<HashMap<DownloadId, Arc<TokioMutex<()>>>>,
}

impl<B: Backend> BackendDownloadManager<B> {
    pub fn from_runtime_handle(runtime_handle: RuntimeHandle) -> Result<Self, DownloadError> {
        let context = B::create_context(runtime_handle)?;
        let manager_id = generate_manager_id(B::manager_suffix());
        log(DownloadLogEvent::ManagerCreated {
            manager_id: manager_id.clone(),
        });
        Ok(Self {
            manager_id,
            instance_id: Uuid::new_v4(),
            context: Arc::new(context),
            tasks: TokioMutex::new(HashMap::new()),
            construction_locks: TokioMutex::new(HashMap::new()),
        })
    }

    async fn cached_task(
        &self,
        request: &DownloadTaskRequest,
    ) -> Option<Result<Arc<DownloadTask>, DownloadError>> {
        let tasks = self.tasks.lock().await;
        let cached = tasks.get(&request.download_id())?;
        let task = cached.task.upgrade()?;
        Some(if task.request() == request {
            Ok(task)
        } else {
            Err(DownloadError::ConflictingConfig(request.destination.display().to_string()))
        })
    }

    async fn stopping_actor(
        &self,
        request: &DownloadTaskRequest,
    ) -> Option<TokioWatchReceiver<DownloadState>> {
        self.tasks.lock().await.get(&request.download_id()).and_then(|cached| cached.actor.clone())
    }

    async fn build_task(
        &self,
        request: &DownloadTaskRequest,
    ) -> Result<(DownloadTask, Option<TokioWatchReceiver<DownloadState>>), DownloadError> {
        let download_id = request.download_id();
        match &request.kind {
            DownloadTaskKind::File {
                source_url,
                file_check,
                expected_bytes,
            } => {
                let config = Arc::new(DownloadConfig {
                    download_id,
                    source_url: source_url.clone(),
                    destination: request.destination.clone(),
                    file_check: file_check.clone(),
                    expected_bytes: *expected_bytes,
                    manager_id: self.manager_id.clone(),
                    manager_instance_id: self.instance_id,
                });
                let (startup, startup_lease) = self.prepare_startup(Startup::observe::<B>(config).await?).await?;
                log(DownloadLogEvent::StartupReconciled {
                    download_id,
                    initial_lifecycle_state: startup.decision.initial_lifecycle_state.name(),
                    action_count: startup.decision.action_plan.as_slice().len(),
                });
                let (task, actor) = spawn_file_download_task::<B>(
                    request.clone(),
                    startup.config,
                    Arc::clone(&self.context),
                    startup.decision,
                    startup_lease,
                )
                .await?;
                log(DownloadLogEvent::TaskSpawned {
                    download_id,
                });
                Ok((DownloadTask::File(task), Some(actor)))
            },
            DownloadTaskKind::Group(subrequests) => {
                let mut children = Vec::with_capacity(subrequests.len());
                for subrequest in subrequests {
                    children.push(self.download_task(subrequest.clone()).await?);
                }
                Ok((DownloadTask::Group(GroupDownloadTask::new(request.clone(), children)), None))
            },
        }
    }

    async fn prepare_startup(
        &self,
        startup: Startup,
    ) -> Result<(Startup, Option<DestinationLockLease>), DownloadError> {
        let attach = startup_may_attach_initial_task::<B>(&startup);
        if startup.lock_state.is_conflict() || (startup.decision.action_plan.is_empty() && !attach) {
            return Ok((startup, None));
        }

        let lease = match DestinationLockLease::acquire_for_destination(
            &startup.config.destination,
            &self.manager_id,
            self.instance_id,
        )
        .await
        {
            Ok(lease) => lease,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                let startup = Startup::observe::<B>(startup.config).await?;
                if startup.lock_state.is_conflict() {
                    return Ok((startup, None));
                }
                return Err(DownloadError::from(error));
            },
            Err(error) => return Err(DownloadError::from(error)),
        };
        let startup = match Startup::observe::<B>(startup.config).await {
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

        if attach && !matches!(startup.decision.initial_lifecycle_state, InitialLifecycleState::Downloaded) {
            Ok((startup, Some(lease)))
        } else {
            lease.release().await?;
            Ok((startup, None))
        }
    }

    async fn construction_lock(
        &self,
        download_id: DownloadId,
    ) -> Arc<TokioMutex<()>> {
        let mut construction_locks = self.construction_locks.lock().await;
        construction_locks.entry(download_id).or_insert_with(|| Arc::new(TokioMutex::new(()))).clone()
    }

    async fn remove_construction_lock_if_unshared(
        &self,
        download_id: DownloadId,
        construction_lock: &Arc<TokioMutex<()>>,
    ) {
        let mut construction_locks = self.construction_locks.lock().await;
        let Some(cached_lock) = construction_locks.get(&download_id) else {
            return;
        };
        if Arc::ptr_eq(cached_lock, construction_lock) && Arc::strong_count(construction_lock) == 2 {
            construction_locks.remove(&download_id);
        }
    }
}

#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
impl<B: Backend> DownloadManager for BackendDownloadManager<B> {
    fn manager_id(&self) -> &str {
        &self.manager_id
    }

    async fn download_task(
        &self,
        request: DownloadTaskRequest,
    ) -> Result<Arc<DownloadTask>, DownloadError> {
        let request = resolve(Path::new(""), request);
        let download_id = request.download_id();
        if let Some(cached) = self.cached_task(&request).await {
            return cached;
        }

        let construction_lock = self.construction_lock(download_id).await;
        let _construction_guard = construction_lock.lock().await;
        let result = async {
            if let Some(cached) = self.cached_task(&request).await {
                return cached;
            }
            if let Some(mut actor) = self.stopping_actor(&request).await {
                while actor.changed().await.is_ok() {}
            }
            let (task, actor) = self.build_task(&request).await?;
            let task = Arc::new(task);
            let mut tasks = self.tasks.lock().await;
            tasks.retain(|_, cached| !cached.is_stopped());
            tasks.insert(
                download_id,
                CachedDownloadTask {
                    task: Arc::downgrade(&task),
                    actor,
                },
            );
            Ok(task)
        }
        .await;
        self.remove_construction_lock_if_unshared(download_id, &construction_lock).await;
        result
    }
}

fn resolve(
    parent: &Path,
    mut request: DownloadTaskRequest,
) -> DownloadTaskRequest {
    request.destination = parent.join(&request.destination);
    if let DownloadTaskKind::Group(subrequests) = &mut request.kind {
        for subrequest in subrequests.iter_mut() {
            let resolved = resolve(&request.destination, std::mem::replace(subrequest, placeholder()));
            *subrequest = resolved;
        }
    }
    request
}

fn placeholder() -> DownloadTaskRequest {
    DownloadTaskRequest::group().destination(PathBuf::new()).subrequests(Vec::new()).build()
}

fn startup_may_attach_initial_task<B: Backend>(startup: &Startup) -> bool {
    B::SUPPORTS_INITIAL_TASK_ATTACHMENT
        && matches!(startup.lock_state, LockFileState::OwnedBySameAppOldProcess(_))
        && !matches!(startup.decision.initial_lifecycle_state, InitialLifecycleState::Downloaded)
}

fn generate_manager_id(suffix: &str) -> String {
    #[cfg(target_vendor = "apple")]
    {
        use objc2_foundation::NSBundle;

        let bundle_id = NSBundle::mainBundle().bundleIdentifier().unwrap_or_default().to_string();
        if bundle_id.is_empty() {
            format!("mirai.{suffix}")
        } else {
            format!("{bundle_id}.mirai.{suffix}")
        }
    }

    #[cfg(not(target_vendor = "apple"))]
    {
        format!("mirai.{suffix}")
    }
}
