use std::{fmt::Display, sync::Arc, time::Duration};

use kiban::{fs, rt};
use tokio::sync::{
    mpsc::{Receiver as TokioMpscReceiver, channel as tokio_mpsc_channel},
    oneshot::Sender as TokioOneshotSender,
    watch::{Receiver as TokioWatchReceiver, Sender as TokioWatchSender, channel as tokio_watch_channel},
};

use crate::{
    DownloadError, DownloadState, DownloadTaskRequest,
    backends::{Backend, BackendEvent, BackendEventSender, BackendProgress, DownloadGeneration},
    file_download::{Command, DownloadConfig, FileDownloadTask, Lifecycle},
    locks::{DestinationLock, LockError},
};

const OBSERVE_INTERVAL: Duration = Duration::from_secs(1);

pub struct FileDownloadActor {
    backend: Arc<dyn Backend>,
    config: Arc<DownloadConfig>,
    lifecycle: Lifecycle,
    generation: DownloadGeneration,
    wants_download: bool,
    events: BackendEventSender,
    commands: TokioMpscReceiver<(Command, TokioOneshotSender<Result<(), DownloadError>>)>,
    backend_events: TokioMpscReceiver<BackendEvent>,
    backend_progress: TokioWatchReceiver<Option<BackendProgress>>,
    state: TokioWatchSender<DownloadState>,
}

impl FileDownloadActor {
    pub async fn spawn(
        backend: Arc<dyn Backend>,
        request: DownloadTaskRequest,
        config: Arc<DownloadConfig>,
        lifecycle: Lifecycle,
        attach_lock: Option<DestinationLock>,
    ) -> Result<FileDownloadTask, DownloadError> {
        let (command_sender, commands) = tokio_mpsc_channel(64);
        let (terminal_sender, backend_events) = tokio_mpsc_channel(64);
        let (progress_sender, backend_progress) = tokio_watch_channel(None);
        let (state, state_receiver) = tokio_watch_channel(lifecycle.download_state(&config));
        let mut actor = Self {
            backend,
            config: Arc::clone(&config),
            lifecycle,
            generation: DownloadGeneration::default(),
            wants_download: false,
            events: BackendEventSender::new(config.download_id, terminal_sender, progress_sender),
            commands,
            backend_events,
            backend_progress,
            state,
        };
        if let Some(lock) = attach_lock {
            actor.attach(lock).await?;
        }
        actor.publish();
        rt::spawn(actor.run());
        Ok(FileDownloadTask::new(request, config, command_sender, state_receiver))
    }

    async fn run(mut self) {
        loop {
            tokio::select! {
                command = self.commands.recv() => {
                    let Some((command, reply)) = command else { break };
                    let result = match command {
                        Command::Download => self.download().await,
                        Command::Pause => self.pause().await,
                        Command::Delete => self.delete().await,
                    };
                    self.publish();
                    let _ = reply.send(result);
                },
                event = self.backend_events.recv() => {
                    let Some(event) = event else { break };
                    self.on_backend_event(event).await;
                    self.publish();
                },
                changed = self.backend_progress.changed() => {
                    if changed.is_err() {
                        break;
                    }
                    self.on_backend_progress();
                    self.publish();
                },
                _ = kiban::time::sleep(OBSERVE_INTERVAL), if matches!(self.lifecycle, Lifecycle::Locked { .. }) => {
                    self.observe().await;
                    self.publish();
                },
            }
        }
        if self.lifecycle.is_downloading() {
            let _ = self.pause().await;
            self.publish();
        }
    }

    async fn download(&mut self) -> Result<(), DownloadError> {
        self.wants_download = true;
        match self.lifecycle {
            Lifecycle::NotDownloaded
            | Lifecycle::Paused {
                ..
            }
            | Lifecycle::Failed {
                ..
            } => self.start().await,
            Lifecycle::Downloading {
                ..
            }
            | Lifecycle::Downloaded {
                ..
            }
            | Lifecycle::Locked {
                ..
            } => Ok(()),
        }
    }

    async fn start(&mut self) -> Result<(), DownloadError> {
        let lock = match self.lock().await {
            Ok(lock) => lock,
            Err(DownloadError::LockedByOther(_)) => return Ok(()),
            Err(error) => return Err(error),
        };
        let generation = self.generation.advance();
        let downloaded_bytes = self.backend.read_resume_progress(&self.config.resume_artifact_path).await;
        match self.backend.start(Arc::clone(&self.config), generation, self.events.clone()).await {
            Ok(active_task) => {
                self.lifecycle = Lifecycle::Downloading {
                    active_task,
                    lock,
                    downloaded_bytes,
                    total_bytes: None,
                };
                Ok(())
            },
            Err(error) => {
                let _ = fs::asyn::remove_file(&self.config.resume_artifact_path).await;
                Err(self.fail(error))
            },
        }
    }

    async fn pause(&mut self) -> Result<(), DownloadError> {
        self.wants_download = false;
        match std::mem::replace(&mut self.lifecycle, Lifecycle::NotDownloaded) {
            Lifecycle::Downloading {
                active_task,
                lock: _lock,
                ..
            } => {
                if let Err(error) = active_task.pause(&self.config.resume_artifact_path).await {
                    return Err(self.fail(error));
                }
                if fs::asyn::is_file(&self.config.destination).await {
                    self.complete().await;
                } else {
                    self.lifecycle = Lifecycle::Paused {
                        downloaded_bytes: self.backend.read_resume_progress(&self.config.resume_artifact_path).await,
                    };
                }
                Ok(())
            },
            other => {
                self.lifecycle = other;
                Ok(())
            },
        }
    }

    async fn delete(&mut self) -> Result<(), DownloadError> {
        self.wants_download = false;
        let lock = match std::mem::replace(&mut self.lifecycle, Lifecycle::NotDownloaded) {
            Lifecycle::Downloading {
                active_task,
                lock,
                ..
            } => {
                active_task.cancel().await;
                lock
            },
            other => {
                self.lifecycle = other;
                self.lock().await?
            },
        };
        self.backend.remove_files(&self.config).await;
        lock.remove().await;
        self.lifecycle = Lifecycle::NotDownloaded;
        Ok(())
    }

    async fn lock(&mut self) -> Result<DestinationLock, DownloadError> {
        match self.backend.lock(&self.config).await {
            Ok(lock) => Ok(lock),
            Err(LockError::LockedByOther {
                manager_id,
            }) => {
                let downloaded_bytes = self.backend.read_resume_progress(&self.config.resume_artifact_path).await;
                self.lifecycle = Lifecycle::Locked {
                    manager_id: manager_id.clone(),
                    downloaded_bytes,
                };
                Err(DownloadError::LockedByOther(manager_id))
            },
            Err(error) => Err(error.into()),
        }
    }

    async fn attach(
        &mut self,
        lock: DestinationLock,
    ) -> Result<(), DownloadError> {
        let generation = self.generation.advance();
        if let Some(active_task) =
            self.backend.attach_pending_task(Arc::clone(&self.config), generation, self.events.clone()).await?
        {
            self.lifecycle = Lifecycle::Downloading {
                active_task,
                lock,
                downloaded_bytes: 0,
                total_bytes: None,
            };
        }
        Ok(())
    }

    async fn complete(&mut self) {
        self.lifecycle = match self.backend.verify(&self.config).await {
            Ok(total_bytes) => {
                let _ = fs::asyn::remove_file(&self.config.resume_artifact_path).await;
                Lifecycle::Downloaded {
                    total_bytes,
                }
            },
            Err(message) => {
                self.backend.remove_files(&self.config).await;
                Lifecycle::Failed {
                    message,
                }
            },
        };
    }

    async fn on_backend_event(
        &mut self,
        event: BackendEvent,
    ) {
        if event.generation() != self.generation || !self.lifecycle.is_downloading() {
            return;
        }
        let Lifecycle::Downloading {
            active_task,
            lock: _lock,
            ..
        } = std::mem::replace(&mut self.lifecycle, Lifecycle::NotDownloaded)
        else {
            return;
        };
        match event {
            BackendEvent::Completed {
                ..
            } => self.complete().await,
            BackendEvent::Error {
                message,
                ..
            } => {
                active_task.cancel().await;
                let _ = fs::asyn::remove_file(&self.config.resume_artifact_path).await;
                self.lifecycle = Lifecycle::Failed {
                    message,
                };
            },
        }
    }

    fn on_backend_progress(&mut self) {
        let Some(progress) = *self.backend_progress.borrow_and_update() else {
            return;
        };
        if progress.generation == self.generation
            && let Lifecycle::Downloading {
                downloaded_bytes,
                total_bytes,
                ..
            } = &mut self.lifecycle
        {
            *downloaded_bytes = progress.downloaded_bytes;
            *total_bytes = progress.total_bytes;
        }
    }

    async fn observe(&mut self) {
        match self.backend.reconcile(&self.config).await {
            Ok((lifecycle, lock)) => {
                self.lifecycle = lifecycle;
                if let Some(lock) = lock
                    && let Err(error) = self.attach(lock).await
                {
                    self.fail(error);
                }
                if self.wants_download && matches!(self.lifecycle, Lifecycle::NotDownloaded | Lifecycle::Paused { .. })
                {
                    let _ = self.start().await;
                }
            },
            Err(error) => {
                self.fail(error);
            },
        }
    }

    fn fail(
        &mut self,
        error: impl Display,
    ) -> DownloadError {
        let message = error.to_string();
        self.lifecycle = Lifecycle::Failed {
            message: message.clone(),
        };
        DownloadError::Backend(message)
    }

    fn publish(&self) {
        let next = self.lifecycle.download_state(&self.config);
        self.state.send_if_modified(|current| {
            if *current == next {
                return false;
            }
            if current.phase != next.phase {
                tracing::debug!(
                    download_id = %self.config.download_id,
                    from = ?current.phase,
                    to = ?next.phase,
                    "download phase changed"
                );
            }
            *current = next;
            true
        });
    }
}
