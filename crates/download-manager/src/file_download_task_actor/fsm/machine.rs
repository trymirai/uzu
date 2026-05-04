use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use statig::{
    Outcome::{self, *},
    awaitable::{IntoStateMachineExt as _, StateMachine},
    state_machine,
};

use crate::{
    DownloadError, FileCheck, LockFileState, acquire_lock, check_lock_file,
    crc_utils::{calculate_and_verify_crc, crc_path_for_file, save_crc_file},
    file_download_task_actor::{
        ProgressCounters, PublicProjection, TerminalOutcome,
        fsm::{DispatchContext, DownloadActorEffect, FsmEvent},
    },
    release_lock,
    traits::{
        ActiveDownloadGeneration, ActiveDownloadGenerationCounter, ActiveTask, BackendContext, BackendEventSender,
        DownloadBackend, DownloadConfig,
    },
};

#[derive(Debug)]
pub struct DownloadFsm<B: DownloadBackend> {
    config: Arc<DownloadConfig>,
    context: Arc<B::Context>,
    backend_event_sender: BackendEventSender,
    generation_counter: ActiveDownloadGenerationCounter,
}

impl<B: DownloadBackend> DownloadFsm<B> {
    pub fn new(
        config: Arc<DownloadConfig>,
        context: Arc<B::Context>,
        backend_event_sender: BackendEventSender,
    ) -> Self {
        Self {
            config,
            context,
            backend_event_sender,
            generation_counter: ActiveDownloadGenerationCounter::default(),
        }
    }

    pub fn allocate_next_generation(&mut self) -> ActiveDownloadGeneration {
        self.generation_counter.allocate_next()
    }

    pub fn into_state_machine(
        self,
        initial_lifecycle_state: DownloadLifecycleState<B>,
    ) -> StateMachine<Self> {
        let mut state_machine = self.uninitialized_state_machine();
        *state_machine.state_mut() = initial_lifecycle_state;
        state_machine.into()
    }

    async fn acquire_destination_lock(
        &mut self,
        context: &mut DispatchContext<B>,
    ) -> Result<(), DownloadError> {
        let lock_path = lock_path_for_destination(&self.config.destination);
        match check_lock_file(&lock_path, &self.config.manager_id, std::process::id()) {
            LockFileState::OwnedByOtherApp(lock_file_info) => {
                context.push(DownloadActorEffect::SetProjection(PublicProjection::LockedByOther(
                    lock_file_info.manager_id.clone(),
                )));
                Err(DownloadError::LockedByOther(lock_file_info.manager_id))
            },
            LockFileState::Missing
            | LockFileState::OwnedByUs(_)
            | LockFileState::OwnedBySameAppOldProcess(_)
            | LockFileState::Stale(_) => {
                acquire_lock(&lock_path, &self.config.manager_id).await?;
                Ok(())
            },
        }
    }

    async fn release_destination_lock(&self) {
        let _ = release_lock(&lock_path_for_destination(&self.config.destination)).await;
    }

    async fn start_fresh_download(
        &mut self,
        context: &mut DispatchContext<B>,
    ) -> Outcome<DownloadLifecycleState<B>> {
        if let Err(error) = self.acquire_destination_lock(context).await {
            context.reply(Err(error));
            return Handled;
        }

        let generation = self.allocate_next_generation();
        let active_task = match self
            .context
            .download(Arc::clone(&self.config), generation, self.backend_event_sender.clone())
            .await
        {
            Ok(active_task) => active_task,
            Err(error) => {
                self.release_destination_lock().await;
                context.reply(Err(DownloadError::Backend(error.to_string())));
                return Handled;
            },
        };

        context.push(DownloadActorEffect::SetProjection(PublicProjection::None));
        context.push(DownloadActorEffect::SetProgress(ProgressCounters {
            downloaded_bytes: 0,
            total_bytes: self.config.expected_bytes.unwrap_or(0),
        }));
        context.reply(Ok(()));
        Transition(DownloadLifecycleState::downloading(Some(active_task), generation))
    }

    async fn resume_download(
        &mut self,
        part_path: PathBuf,
        context: &mut DispatchContext<B>,
    ) -> Outcome<DownloadLifecycleState<B>> {
        if !part_path.exists() {
            remove_file(&part_path);
            let message = "resume artifact is missing".to_string();
            context.push(DownloadActorEffect::SetProjection(PublicProjection::StickyError(message.clone())));
            context.push(DownloadActorEffect::SetProgress(ProgressCounters::default()));
            context.push(DownloadActorEffect::CompleteWaiters(TerminalOutcome::Error(message.clone())));
            context.reply(Err(DownloadError::Backend(message)));
            return Transition(DownloadLifecycleState::not_downloaded());
        }

        if let Err(error) = self.acquire_destination_lock(context).await {
            context.reply(Err(error));
            return Handled;
        }

        let generation = self.allocate_next_generation();
        let resume_bytes = file_size(&part_path).unwrap_or(0);
        let active_task = match self
            .context
            .resume(Arc::clone(&self.config), generation, &part_path, self.backend_event_sender.clone())
            .await
        {
            Ok(active_task) => active_task,
            Err(error) => {
                self.release_destination_lock().await;
                remove_file(&part_path);
                let message = error.to_string();
                context.push(DownloadActorEffect::SetProjection(PublicProjection::StickyError(message.clone())));
                context.push(DownloadActorEffect::SetProgress(ProgressCounters::default()));
                context.push(DownloadActorEffect::CompleteWaiters(TerminalOutcome::Error(message.clone())));
                context.reply(Err(DownloadError::Backend(message)));
                return Transition(DownloadLifecycleState::not_downloaded());
            },
        };

        context.push(DownloadActorEffect::SetProjection(PublicProjection::None));
        context.push(DownloadActorEffect::SetProgress(ProgressCounters {
            downloaded_bytes: resume_bytes,
            total_bytes: self.config.expected_bytes.unwrap_or(resume_bytes),
        }));
        context.reply(Ok(()));
        Transition(DownloadLifecycleState::downloading(Some(active_task), generation))
    }

    fn active_task_missing(
        &self,
        context: &mut DispatchContext<B>,
    ) -> Outcome<DownloadLifecycleState<B>> {
        let message = "active task missing while downloading".to_string();
        tracing::error!(message, "download FSM invariant failed");
        context.push(DownloadActorEffect::SetProjection(PublicProjection::StickyError(message.clone())));
        context.push(DownloadActorEffect::SetProgress(ProgressCounters::default()));
        context.push(DownloadActorEffect::CompleteWaiters(TerminalOutcome::Error(message)));
        Transition(DownloadLifecycleState::not_downloaded())
    }

    async fn handle_completion(
        &mut self,
        context: &mut DispatchContext<B>,
    ) -> Outcome<DownloadLifecycleState<B>> {
        match validate_completed_file(&self.config).await {
            Ok(total_bytes) => {
                context.push(DownloadActorEffect::SetProgress(ProgressCounters {
                    downloaded_bytes: total_bytes,
                    total_bytes,
                }));
                context.push(DownloadActorEffect::SetProjection(PublicProjection::None));
                context.push(DownloadActorEffect::CompleteWaiters(TerminalOutcome::Downloaded));
                Transition(DownloadLifecycleState::downloaded(
                    self.config.destination.clone(),
                    crc_path_for_destination(&self.config.destination),
                ))
            },
            Err(message) => {
                remove_file(&self.config.destination);
                remove_resume_artifact(&self.config.destination);
                context.push(DownloadActorEffect::SetProgress(ProgressCounters::default()));
                context.push(DownloadActorEffect::SetProjection(PublicProjection::StickyError(message.clone())));
                context.push(DownloadActorEffect::CompleteWaiters(TerminalOutcome::Error(message)));
                Transition(DownloadLifecycleState::not_downloaded())
            },
        }
    }
}

#[state_machine(
    initial = "DownloadLifecycleState::not_downloaded()",
    state(name = "DownloadLifecycleState", derive(Debug)),
    superstate(name = "DownloadSuperstate", derive(Debug)),
    after_transition = "Self::log_transition"
)]
impl<B: DownloadBackend> DownloadFsm<B> {
    #[superstate]
    async fn idle(
        &mut self,
        context: &mut DispatchContext<B>,
        event: &FsmEvent,
    ) -> Outcome<DownloadLifecycleState<B>> {
        match event {
            FsmEvent::Pause => {
                context.reply(Err(DownloadError::InvalidStateTransition));
                Handled
            },
            FsmEvent::Cancel => {
                context.reply(Ok(()));
                Handled
            },
            _ => Super,
        }
    }

    #[state(superstate = "idle")]
    async fn not_downloaded(
        &mut self,
        context: &mut DispatchContext<B>,
        event: &FsmEvent,
    ) -> Outcome<DownloadLifecycleState<B>> {
        match event {
            FsmEvent::Download => self.start_fresh_download(context).await,
            _ => Super,
        }
    }

    #[state(superstate = "idle")]
    async fn downloaded(
        &mut self,
        context: &mut DispatchContext<B>,
        file_path: &mut PathBuf,
        crc_path: &mut Option<PathBuf>,
        event: &FsmEvent,
    ) -> Outcome<DownloadLifecycleState<B>> {
        let _ = (file_path, crc_path);
        match event {
            FsmEvent::Download => {
                context.reply(Ok(()));
                Handled
            },
            _ => Super,
        }
    }

    #[state(exit_action = "download_exited")]
    async fn downloading(
        &mut self,
        context: &mut DispatchContext<B>,
        active_task: &mut Option<B::ActiveTask>,
        generation: &mut ActiveDownloadGeneration,
        event: &FsmEvent,
    ) -> Outcome<DownloadLifecycleState<B>> {
        match event {
            FsmEvent::Pause => {
                let Some(active_task) = active_task.take() else {
                    context.reply(Err(DownloadError::Backend("active task missing while downloading".to_string())));
                    return self.active_task_missing(context);
                };

                match active_task.pause(&self.config.destination).await {
                    Ok(part_path) => {
                        let downloaded_bytes = file_size(&part_path).unwrap_or(0);
                        context.push(DownloadActorEffect::SetProgress(ProgressCounters {
                            downloaded_bytes,
                            total_bytes: self.config.expected_bytes.unwrap_or(downloaded_bytes),
                        }));
                        context.reply(Ok(()));
                        Transition(DownloadLifecycleState::paused(part_path))
                    },
                    Err(error) => {
                        let message = error.to_string();
                        context
                            .push(DownloadActorEffect::SetProjection(PublicProjection::StickyError(message.clone())));
                        context.push(DownloadActorEffect::SetProgress(ProgressCounters::default()));
                        context.push(DownloadActorEffect::CompleteWaiters(TerminalOutcome::Error(message.clone())));
                        context.reply(Err(DownloadError::Backend(message)));
                        Transition(DownloadLifecycleState::not_downloaded())
                    },
                }
            },
            FsmEvent::Cancel => {
                if let Some(active_task) = active_task.take() {
                    let _ = active_task.cancel(&self.config.destination).await;
                }
                remove_resume_artifact(&self.config.destination);
                context.push(DownloadActorEffect::SetProgress(ProgressCounters::default()));
                context.push(DownloadActorEffect::SetProjection(PublicProjection::None));
                context.reply(Ok(()));
                Transition(DownloadLifecycleState::not_downloaded())
            },
            FsmEvent::BackendCompleted {
                generation: event_generation,
            } => {
                if event_generation != generation {
                    return Handled;
                }
                self.handle_completion(context).await
            },
            FsmEvent::BackendError {
                generation: event_generation,
                message,
            } => {
                if event_generation != generation {
                    return Handled;
                }
                if let Some(active_task) = active_task.take() {
                    let _ = active_task.cancel(&self.config.destination).await;
                }
                remove_resume_artifact(&self.config.destination);
                context.push(DownloadActorEffect::SetProjection(PublicProjection::StickyError(message.clone())));
                context.push(DownloadActorEffect::SetProgress(ProgressCounters::default()));
                context.push(DownloadActorEffect::CompleteWaiters(TerminalOutcome::Error(message.clone())));
                Transition(DownloadLifecycleState::not_downloaded())
            },
            FsmEvent::ProgressUpdate {
                generation: event_generation,
                downloaded_bytes,
                total_bytes,
            } => {
                if event_generation == generation {
                    context.push(DownloadActorEffect::SetProgress(ProgressCounters {
                        downloaded_bytes: *downloaded_bytes,
                        total_bytes: total_bytes.or(self.config.expected_bytes).unwrap_or(*downloaded_bytes),
                    }));
                }
                Handled
            },
            FsmEvent::Download => {
                context.reply(Ok(()));
                Handled
            },
        }
    }

    #[action]
    async fn download_exited(&mut self) {
        self.release_destination_lock().await;
    }

    #[state]
    async fn paused(
        &mut self,
        context: &mut DispatchContext<B>,
        part_path: &mut PathBuf,
        event: &FsmEvent,
    ) -> Outcome<DownloadLifecycleState<B>> {
        match event {
            FsmEvent::Download => self.resume_download(part_path.clone(), context).await,
            FsmEvent::Pause => {
                context.reply(Ok(()));
                Handled
            },
            FsmEvent::Cancel => {
                remove_file(part_path);
                context.push(DownloadActorEffect::SetProgress(ProgressCounters::default()));
                context.push(DownloadActorEffect::SetProjection(PublicProjection::None));
                context.reply(Ok(()));
                Transition(DownloadLifecycleState::not_downloaded())
            },
            _ => Handled,
        }
    }

    async fn log_transition(
        &mut self,
        source: &DownloadLifecycleState<B>,
        target: &DownloadLifecycleState<B>,
        context: &mut DispatchContext<B>,
    ) {
        context.push(DownloadActorEffect::LogFsmTransition {
            from: source.name(),
            to: target.name(),
        });
    }
}

impl<B: DownloadBackend> DownloadLifecycleState<B> {
    pub fn name(&self) -> &'static str {
        match self {
            Self::NotDownloaded {} => "NotDownloaded",
            Self::Paused {
                ..
            } => "Paused",
            Self::Downloaded {
                ..
            } => "Downloaded",
            Self::Downloading {
                ..
            } => "Downloading",
        }
    }
}

fn file_size(path: &Path) -> Option<u64> {
    path.metadata().ok().map(|metadata| metadata.len())
}

fn remove_file(path: &Path) {
    let _ = std::fs::remove_file(path);
}

fn remove_resume_artifact(destination: &Path) {
    remove_file(&destination.with_extension("part"));
    remove_file(&destination.with_extension("resume_data"));
}

fn lock_path_for_destination(destination: &Path) -> PathBuf {
    PathBuf::from(format!("{}.lock", destination.display()))
}

fn crc_path_for_destination(destination: &Path) -> Option<PathBuf> {
    let crc_path = crc_path_for_file(destination);
    crc_path.exists().then_some(crc_path)
}

async fn validate_completed_file(config: &DownloadConfig) -> Result<u64, String> {
    let mut metadata = config.destination.metadata().ok();
    for _attempt in 0..10 {
        if metadata.is_some() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        metadata = config.destination.metadata().ok();
    }
    let metadata = metadata.ok_or_else(|| "download completed but destination is missing".to_string())?;
    let total_bytes = config.expected_bytes.unwrap_or(metadata.len());

    match &config.file_check {
        FileCheck::None => Ok(total_bytes),
        FileCheck::CRC(expected_crc) => match calculate_and_verify_crc(&config.destination, expected_crc) {
            Ok(true) => {
                let _ = save_crc_file(&config.destination, expected_crc);
                Ok(total_bytes)
            },
            Ok(false) => Err("CRC verification failed".to_string()),
            Err(error) => Err(format!("CRC verification error: {error}")),
        },
    }
}
