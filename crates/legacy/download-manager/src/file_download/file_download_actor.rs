use std::{sync::Arc, time::Duration};

use kiban::{fs, rt};
use tokio::sync::{
    mpsc::{Receiver as TokioMpscReceiver, channel as tokio_mpsc_channel},
    oneshot::Sender as TokioOneshotSender,
    watch::{Receiver as TokioWatchReceiver, Sender as TokioWatchSender, channel as tokio_watch_channel},
};

use crate::{
    DownloadPhase, DownloadState, DownloadTaskRequest,
    backends::{ActiveTask, Backend, BackendEvent, BackendEventSender, BackendProgress, DownloadGeneration},
    file_download::{Command, DownloadConfig, FileDownloadError, FileDownloadTask},
    locks::{DestinationLock, LockError},
};

const OBSERVE_INTERVAL: Duration = Duration::from_secs(1);

pub struct FileDownloadActor {
    backend: Arc<dyn Backend>,
    config: Arc<DownloadConfig>,
    state: DownloadState,
    active: Option<(Box<dyn ActiveTask>, DestinationLock)>,
    generation: DownloadGeneration,
    wants_download: bool,
    events: BackendEventSender,
    commands: TokioMpscReceiver<(Command, TokioOneshotSender<Result<(), FileDownloadError>>)>,
    backend_events: TokioMpscReceiver<BackendEvent>,
    backend_progress: TokioWatchReceiver<Option<BackendProgress>>,
    published: TokioWatchSender<DownloadState>,
}

impl FileDownloadActor {
    pub async fn spawn(
        backend: Arc<dyn Backend>,
        request: DownloadTaskRequest,
        config: Arc<DownloadConfig>,
        state: DownloadState,
        attach_lock: Option<DestinationLock>,
    ) -> Result<FileDownloadTask, FileDownloadError> {
        let (command_sender, commands) = tokio_mpsc_channel(64);
        let (terminal_sender, backend_events) = tokio_mpsc_channel(64);
        let (progress_sender, backend_progress) = tokio_watch_channel(None);
        let (published, state_receiver) = tokio_watch_channel(state.clone());
        let mut actor = Self {
            backend,
            config: Arc::clone(&config),
            state,
            active: None,
            generation: DownloadGeneration::default(),
            wants_download: false,
            events: BackendEventSender::new(config.download_id, terminal_sender, progress_sender),
            commands,
            backend_events,
            backend_progress,
            published,
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
                _ = kiban::time::sleep(OBSERVE_INTERVAL), if matches!(self.state.phase, DownloadPhase::Locked { .. }) => {
                    self.observe().await;
                    self.publish();
                },
            }
        }
        if self.active.is_some() {
            let _ = self.pause().await;
            self.publish();
        }
    }

    async fn download(&mut self) -> Result<(), FileDownloadError> {
        self.wants_download = true;
        match self.state.phase {
            DownloadPhase::NotDownloaded {}
            | DownloadPhase::Paused {}
            | DownloadPhase::Error {
                ..
            } => self.start().await,
            DownloadPhase::Downloading {}
            | DownloadPhase::Downloaded {}
            | DownloadPhase::Locked {
                ..
            } => Ok(()),
        }
    }

    async fn start(&mut self) -> Result<(), FileDownloadError> {
        let lock = match self.lock().await {
            Ok(lock) => lock,
            Err(FileDownloadError::Lock(LockError::LockedByOther {
                ..
            })) => return Ok(()),
            Err(error) => return Err(error),
        };
        let generation = self.generation.advance();
        let downloaded_bytes = self.backend.read_resume_progress(&self.config.resume_artifact_path).await;
        match self.backend.start(Arc::clone(&self.config), generation, self.events.clone()).await {
            Ok(task) => {
                self.active = Some((task, lock));
                self.set(DownloadPhase::Downloading {}, downloaded_bytes, None);
                Ok(())
            },
            Err(error) => {
                let _ = fs::asyn::remove_file(&self.config.resume_artifact_path).await;
                Err(self.fail(error.into()))
            },
        }
    }

    async fn pause(&mut self) -> Result<(), FileDownloadError> {
        self.wants_download = false;
        let Some((task, _lock)) = self.active.take() else {
            return Ok(());
        };
        if let Err(error) = task.pause(&self.config.resume_artifact_path).await {
            return Err(self.fail(error.into()));
        }
        if fs::asyn::is_file(&self.config.destination).await {
            self.complete().await;
        } else {
            let downloaded_bytes = self.backend.read_resume_progress(&self.config.resume_artifact_path).await;
            self.set(DownloadPhase::Paused {}, downloaded_bytes, None);
        }
        Ok(())
    }

    async fn delete(&mut self) -> Result<(), FileDownloadError> {
        self.wants_download = false;
        let lock = match self.active.take() {
            Some((task, lock)) => {
                task.cancel().await;
                lock
            },
            None => self.lock().await?,
        };
        self.backend.remove_files(&self.config).await;
        lock.remove().await;
        self.set(DownloadPhase::NotDownloaded {}, 0, None);
        Ok(())
    }

    async fn lock(&mut self) -> Result<DestinationLock, FileDownloadError> {
        match self.backend.lock(&self.config).await {
            Ok(lock) => Ok(lock),
            Err(LockError::LockedByOther {
                manager_id,
            }) => {
                let downloaded_bytes = self.backend.read_resume_progress(&self.config.resume_artifact_path).await;
                self.set(
                    DownloadPhase::Locked {
                        manager_id: manager_id.clone(),
                    },
                    downloaded_bytes,
                    None,
                );
                Err(LockError::LockedByOther {
                    manager_id,
                }
                .into())
            },
            Err(error) => Err(error.into()),
        }
    }

    async fn attach(
        &mut self,
        lock: DestinationLock,
    ) -> Result<(), FileDownloadError> {
        let generation = self.generation.advance();
        if let Some(task) =
            self.backend.attach_pending_task(Arc::clone(&self.config), generation, self.events.clone()).await?
        {
            self.active = Some((task, lock));
            self.set(DownloadPhase::Downloading {}, 0, None);
        }
        Ok(())
    }

    async fn complete(&mut self) {
        match self.backend.verify(&self.config).await {
            Ok(total_bytes) => {
                let _ = fs::asyn::remove_file(&self.config.resume_artifact_path).await;
                self.set(DownloadPhase::Downloaded {}, total_bytes, Some(total_bytes));
            },
            Err(error) => {
                self.backend.remove_files(&self.config).await;
                self.set(
                    DownloadPhase::Error {
                        message: error.to_string(),
                    },
                    0,
                    None,
                );
            },
        }
    }

    async fn on_backend_event(
        &mut self,
        event: BackendEvent,
    ) {
        if event.generation() != self.generation {
            return;
        }
        let Some((task, _lock)) = self.active.take() else {
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
                task.cancel().await;
                let _ = fs::asyn::remove_file(&self.config.resume_artifact_path).await;
                self.set(
                    DownloadPhase::Error {
                        message,
                    },
                    0,
                    None,
                );
            },
        }
    }

    fn on_backend_progress(&mut self) {
        let Some(progress) = *self.backend_progress.borrow_and_update() else {
            return;
        };
        if progress.generation == self.generation && self.active.is_some() {
            self.set(DownloadPhase::Downloading {}, progress.downloaded_bytes, progress.total_bytes);
        }
    }

    async fn observe(&mut self) {
        match self.backend.reconcile(&self.config).await {
            Ok((state, lock)) => {
                self.state = state;
                if let Some(lock) = lock
                    && let Err(error) = self.attach(lock).await
                {
                    self.fail(error);
                }
                if self.wants_download
                    && matches!(self.state.phase, DownloadPhase::NotDownloaded {} | DownloadPhase::Paused {})
                {
                    let _ = self.start().await;
                }
            },
            Err(error) => {
                self.fail(error.into());
            },
        }
    }

    fn fail(
        &mut self,
        error: FileDownloadError,
    ) -> FileDownloadError {
        self.set(
            DownloadPhase::Error {
                message: error.to_string(),
            },
            0,
            None,
        );
        error
    }

    fn set(
        &mut self,
        phase: DownloadPhase,
        downloaded_bytes: u64,
        total_bytes: Option<u64>,
    ) {
        self.state = DownloadState::new(&self.config, phase, downloaded_bytes, total_bytes);
    }

    fn publish(&self) {
        self.published.send_if_modified(|current| {
            if *current == self.state {
                return false;
            }
            if current.phase != self.state.phase {
                tracing::debug!(
                    download_id = %self.config.download_id,
                    from = ?current.phase,
                    to = ?self.state.phase,
                    "download phase changed"
                );
            }
            *current = self.state.clone();
            true
        });
    }
}
