use std::sync::Arc;

use statig::awaitable::StateMachine;
use tokio::sync::{
    Mutex as TokioMutex,
    broadcast::Sender as TokioBroadcastSender,
    mpsc::Receiver as TokioMpscReceiver,
    watch::{Receiver as TokioWatchReceiver, Sender as TokioWatchSender},
};

use crate::{
    DownloadError, FileDownloadState,
    download_log_event::{DownloadLogEvent, record_download_log_event},
    file_download_task_actor::{
        BackendEvent, DownloadLifecycleState, PendingProgressSlot, ProgressCounters, PublicProjection, TaskCommand,
        TerminalOutcome,
        fsm::{DispatchContext, DownloadActorEffect, DownloadFsm, FsmEvent},
        project_runtime_public_state,
    },
    lock_manager::lock_path_for_destination,
    release_lock_if_owned,
    traits::{ActiveTask, DownloadBackend, DownloadConfig},
};

enum ActorLoopExit {
    AlreadyStopped,
    PreserveArtifacts,
}

pub struct DownloadTaskActor<B: DownloadBackend> {
    config: Arc<DownloadConfig>,
    lifecycle: StateMachine<DownloadFsm<B>>,
    projection: PublicProjection,
    progress_counters: ProgressCounters,
    command_receiver: TokioMpscReceiver<TaskCommand>,
    backend_event_receiver: TokioMpscReceiver<BackendEvent>,
    pending_progress: Arc<TokioMutex<PendingProgressSlot>>,
    progress_waker_receiver: TokioWatchReceiver<()>,
    public_state_sender: TokioWatchSender<FileDownloadState>,
    progress_sender: TokioBroadcastSender<FileDownloadState>,
    terminal_sender: TokioWatchSender<TerminalOutcome>,
}

impl<B: DownloadBackend> DownloadTaskActor<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: Arc<DownloadConfig>,
        lifecycle: StateMachine<DownloadFsm<B>>,
        projection: PublicProjection,
        progress_counters: ProgressCounters,
        command_receiver: TokioMpscReceiver<TaskCommand>,
        backend_event_receiver: TokioMpscReceiver<BackendEvent>,
        pending_progress: Arc<TokioMutex<PendingProgressSlot>>,
        progress_waker_receiver: TokioWatchReceiver<()>,
        public_state_sender: TokioWatchSender<FileDownloadState>,
        progress_sender: TokioBroadcastSender<FileDownloadState>,
        terminal_sender: TokioWatchSender<TerminalOutcome>,
    ) -> Self {
        Self {
            config,
            lifecycle,
            projection,
            progress_counters,
            command_receiver,
            backend_event_receiver,
            pending_progress,
            progress_waker_receiver,
            public_state_sender,
            progress_sender,
            terminal_sender,
        }
    }

    pub async fn run(mut self) {
        self.publish_current_state();

        let mut loop_exit = ActorLoopExit::PreserveArtifacts;
        loop {
            tokio::select! {
                command = self.command_receiver.recv() => {
                    let Some(command) = command else {
                        break;
                    };
                    if !self.handle_command(command).await {
                        loop_exit = ActorLoopExit::AlreadyStopped;
                        break;
                    }
                }
                backend_event = self.backend_event_receiver.recv() => {
                    let Some(backend_event) = backend_event else {
                        break;
                    };
                    self.handle_backend_event(backend_event).await;
                }
                progress_wake_result = self.progress_waker_receiver.changed() => {
                    if progress_wake_result.is_err() {
                        break;
                    }
                    self.handle_pending_progress().await;
                }
            }
        }

        if matches!(loop_exit, ActorLoopExit::PreserveArtifacts) {
            self.stop_preserving_artifacts().await;
        }
        let _ = self.terminal_sender.send(TerminalOutcome::ActorStopped);
    }

    pub(crate) async fn abort_before_start(mut self) {
        // SAFETY: The actor was never scheduled, so no state handler can be active.
        // We only take the backend task before discarding the whole state machine.
        let lifecycle_state = unsafe { self.lifecycle.state_mut() };
        let active_task = match lifecycle_state {
            DownloadLifecycleState::Downloading {
                active_task,
                ..
            } => active_task.take(),
            _ => None,
        };
        if let Some(active_task) = active_task {
            let _ = active_task.cancel(&self.config.destination).await;
        }
        let lock_path = lock_path_for_destination(&self.config.destination);
        let _ = release_lock_if_owned(&lock_path, &self.config.manager_id, self.config.manager_instance_id).await;
        let _ = self.terminal_sender.send(TerminalOutcome::ActorStopped);
    }

    async fn stop_preserving_artifacts(&mut self) {
        self.dispatch(FsmEvent::StopPreservingArtifacts, None).await;
        let lock_path = lock_path_for_destination(&self.config.destination);
        let _ = release_lock_if_owned(&lock_path, &self.config.manager_id, self.config.manager_instance_id).await;
    }

    async fn handle_command(
        &mut self,
        command: TaskCommand,
    ) -> bool {
        match command {
            TaskCommand::Download {
                reply_sender,
            } => {
                self.dispatch(FsmEvent::Download, Some(reply_sender)).await;
                true
            },
            TaskCommand::Pause {
                reply_sender,
            } => {
                self.dispatch(FsmEvent::Pause, Some(reply_sender)).await;
                true
            },
            TaskCommand::Cancel {
                reply_sender,
            } => {
                self.dispatch(FsmEvent::Cancel, Some(reply_sender)).await;
                true
            },
            TaskCommand::Remove {
                reply_sender,
            } => {
                self.dispatch(FsmEvent::Remove, Some(reply_sender)).await;
                false
            },
        }
    }

    async fn handle_backend_event(
        &mut self,
        backend_event: BackendEvent,
    ) {
        match backend_event {
            BackendEvent::Completed {
                generation,
            } => {
                self.dispatch(
                    FsmEvent::BackendCompleted {
                        generation,
                    },
                    None,
                )
                .await
            },
            BackendEvent::Error {
                message,
                generation,
            } => {
                self.dispatch(
                    FsmEvent::BackendError {
                        generation,
                        message,
                    },
                    None,
                )
                .await
            },
        }
    }

    async fn handle_pending_progress(&mut self) {
        let progress = self.pending_progress.lock().await.take();
        let Some(progress) = progress else {
            return;
        };

        self.dispatch(
            FsmEvent::ProgressUpdate {
                generation: progress.generation,
                downloaded_bytes: progress.downloaded_bytes,
                total_bytes: progress.total_bytes,
            },
            None,
        )
        .await;
    }

    fn publish_current_state(&self) {
        let public_state = project_runtime_public_state(
            self.lifecycle.state(),
            &self.projection,
            self.progress_counters,
            &self.config,
        );
        let _ = self.public_state_sender.send(public_state.clone());
        let _ = self.progress_sender.send(public_state);
    }

    async fn dispatch(
        &mut self,
        event: FsmEvent,
        reply_sender: Option<tokio::sync::oneshot::Sender<Result<(), DownloadError>>>,
    ) {
        let mut dispatch_context = DispatchContext::new(reply_sender);
        self.lifecycle.handle_with_context(&event, &mut dispatch_context).await;
        let pending_reply = self.apply_non_reply_effects(&mut dispatch_context).await;
        self.publish_current_state();
        if let Some((reply_sender, result)) = pending_reply {
            let _ = reply_sender.send(result);
        }
    }

    async fn apply_non_reply_effects(
        &mut self,
        dispatch_context: &mut DispatchContext,
    ) -> Option<(tokio::sync::oneshot::Sender<Result<(), DownloadError>>, Result<(), DownloadError>)> {
        let mut pending_reply = None;
        for effect in dispatch_context.effects.drain(..) {
            match effect {
                DownloadActorEffect::SetProjection(projection) => self.projection = projection,
                DownloadActorEffect::SetProgress(progress_counters) => self.progress_counters = progress_counters,
                DownloadActorEffect::CompleteWaiters(terminal_outcome) => {
                    let _ = self.terminal_sender.send(terminal_outcome);
                },
                DownloadActorEffect::Reply(result) => {
                    if let Some(reply_sender) = dispatch_context.pending_reply.take() {
                        pending_reply = Some((reply_sender, result));
                    }
                },
                DownloadActorEffect::LogFsmTransition {
                    from,
                    to,
                } => {
                    record_download_log_event(DownloadLogEvent::FsmTransition {
                        from,
                        to,
                    });
                },
            }
        }
        pending_reply
    }
}
