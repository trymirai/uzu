use std::{future::Future, pin::Pin, sync::OnceLock};

use tokio::{
    runtime::Builder as TokioRuntimeBuilder,
    sync::mpsc::{UnboundedSender, unbounded_channel},
    task::LocalSet,
};

use crate::{DownloadError, file_download_task_actor::DownloadTaskActor, traits::DownloadBackend};

type SpawnJobBox = Box<dyn SpawnJob>;

trait SpawnJob: Send {
    fn spawn(self: Box<Self>);
    fn abort_before_start(self: Box<Self>) -> Pin<Box<dyn Future<Output = ()> + Send>>;
}

struct ActorSpawnJob<B: DownloadBackend>
where
    DownloadTaskActor<B>: Send + 'static,
{
    actor: DownloadTaskActor<B>,
}

impl<B: DownloadBackend> SpawnJob for ActorSpawnJob<B>
where
    DownloadTaskActor<B>: Send + 'static,
{
    fn spawn(self: Box<Self>) {
        tokio::task::spawn_local(self.actor.run());
    }

    fn abort_before_start(self: Box<Self>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        Box::pin(async move {
            self.actor.abort_before_start().await;
        })
    }
}

pub(crate) struct ActorSpawnError {
    spawn_job: SpawnJobBox,
    error: DownloadError,
}

impl ActorSpawnError {
    pub(crate) async fn abort_actor_before_start(self) -> DownloadError {
        self.spawn_job.abort_before_start().await;
        self.error
    }

    pub(crate) fn into_download_error(self) -> DownloadError {
        self.error
    }
}

struct LocalActorScheduler {
    sender: UnboundedSender<SpawnJobBox>,
}

impl LocalActorScheduler {
    fn new() -> Result<Self, DownloadError> {
        let (sender, mut receiver) = unbounded_channel::<SpawnJobBox>();
        let (startup_sender, startup_receiver) = std::sync::mpsc::channel();
        let _scheduler_thread = std::thread::Builder::new()
            .name("download-manager-actor-scheduler".to_string())
            .spawn(move || {
                let runtime = match TokioRuntimeBuilder::new_current_thread().enable_all().build() {
                    Ok(runtime) => runtime,
                    Err(error) => {
                        let startup_error = DownloadError::ActorSchedulerStartFailed(error.to_string());
                        let _ = startup_sender.send(Err(startup_error));
                        return;
                    },
                };
                let local_set = LocalSet::new();
                let _ = startup_sender.send(Ok(()));
                local_set.block_on(&runtime, async move {
                    while let Some(spawn_job) = receiver.recv().await {
                        spawn_job.spawn();
                    }
                });
            })
            .map_err(|error| DownloadError::ActorSchedulerStartFailed(error.to_string()))?;
        let startup_result =
            startup_receiver.recv().map_err(|error| DownloadError::ActorSchedulerStartFailed(error.to_string()))?;
        startup_result?;
        Ok(Self {
            sender,
        })
    }
}

fn scheduler() -> Result<&'static LocalActorScheduler, DownloadError> {
    static SCHEDULER: OnceLock<LocalActorScheduler> = OnceLock::new();
    if let Some(scheduler) = SCHEDULER.get() {
        return Ok(scheduler);
    }

    let scheduler = LocalActorScheduler::new()?;
    let _ = SCHEDULER.set(scheduler);
    SCHEDULER.get().ok_or_else(|| {
        DownloadError::ActorSchedulerStartFailed("scheduler initialization did not complete".to_string())
    })
}

// All download actors share a single dedicated thread that runs a
// `current_thread` Tokio runtime and a `LocalSet`. Each actor is `spawn_local`'d
// onto that LocalSet, so they cooperate on one thread instead of pinning one
// blocking-pool thread per active download. `spawn_local` does not require the
// future to be `Send`, which sidesteps the HRTB issues that prevented us from
// using plain `tokio::spawn` (statig's awaitable handlers and the actor's
// select-arm receiver borrows produce futures that aren't `Send` for all
// lifetimes).
//
// The actor itself must be `Send` to cross the channel into the scheduler
// thread, which it already is (the same constraint `spawn_blocking` had).
pub(crate) fn spawn_actor<B: DownloadBackend>(actor: DownloadTaskActor<B>) -> Result<(), ActorSpawnError>
where
    DownloadTaskActor<B>: Send + 'static,
{
    let spawn_job: SpawnJobBox = Box::new(ActorSpawnJob {
        actor,
    });
    let scheduler = match scheduler() {
        Ok(scheduler) => scheduler,
        Err(error) => {
            return Err(ActorSpawnError {
                spawn_job,
                error,
            });
        },
    };
    scheduler.sender.send(spawn_job).map_err(|error| ActorSpawnError {
        spawn_job: error.0,
        error: DownloadError::ActorSchedulerStopped,
    })
}
