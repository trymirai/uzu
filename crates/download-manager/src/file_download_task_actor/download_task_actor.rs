use std::sync::Arc;

use statig::awaitable::StateMachine;
use tokio::sync::{
    Mutex as TokioMutex,
    broadcast::Sender as TokioBroadcastSender,
    mpsc::Receiver as TokioMpscReceiver,
    watch::{Receiver as TokioWatchReceiver, Sender as TokioWatchSender},
};

use crate::{
    DownloadError, DownloadLogEvent, FileDownloadState,
    file_download_task_actor::{
        BackendEvent, PendingProgressSlot, ProgressCounters, PublicProjection, TaskCommand, TerminalOutcome,
        fsm::{DispatchContext, DownloadActorEffect, DownloadFsm, FsmEvent},
        project_runtime_public_state,
    },
    record_download_log_event,
    traits::{DownloadBackend, DownloadConfig},
};

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

        loop {
            tokio::select! {
                command = self.command_receiver.recv() => {
                    let Some(command) = command else {
                        break;
                    };
                    self.handle_command(command).await;
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

        let _ = self.terminal_sender.send(TerminalOutcome::ActorStopped);
    }

    async fn handle_command(
        &mut self,
        command: TaskCommand,
    ) {
        match command {
            TaskCommand::Download {
                reply_sender,
            } => {
                self.dispatch(FsmEvent::Download, Some(reply_sender)).await;
            },
            TaskCommand::Pause {
                reply_sender,
            } => {
                self.dispatch(FsmEvent::Pause, Some(reply_sender)).await;
            },
            TaskCommand::Cancel {
                reply_sender,
            } => {
                self.dispatch(FsmEvent::Cancel, Some(reply_sender)).await;
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
        let mut dispatch_context = DispatchContext::<B>::new(reply_sender);
        self.lifecycle.handle_with_context(&event, &mut dispatch_context).await;
        let pending_reply = self.apply_non_reply_effects(&mut dispatch_context).await;
        self.publish_current_state();
        if let Some((reply_sender, result)) = pending_reply {
            let _ = reply_sender.send(result);
        }
    }

    async fn apply_non_reply_effects(
        &mut self,
        dispatch_context: &mut DispatchContext<B>,
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
                DownloadActorEffect::DeleteFile {
                    path,
                } => {
                    let _ = std::fs::remove_file(path);
                },
                DownloadActorEffect::DeleteResumeArtifacts {
                    destination,
                } => {
                    let _ = std::fs::remove_file(destination.with_extension("part"));
                    let _ = std::fs::remove_file(destination.with_extension("resume_data"));
                },
                DownloadActorEffect::EmitGlobalEvent(event) => {
                    record_download_log_event(DownloadLogEvent::PublicEventEmitted {
                        download_id: self.config.download_id,
                        event,
                    });
                },
                DownloadActorEffect::AttachActiveTask(_) => {},
            }
        }
        pending_reply
    }
}
