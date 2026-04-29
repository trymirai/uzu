use std::{fmt::Debug, future::Future, marker::PhantomData, path::PathBuf};

use async_trait::async_trait;
use statig::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ActiveDownloadGeneration(u64);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResumeArtifactPath(PathBuf);

pub trait DownloadBackend: Clone + Debug + Send + Sync + 'static {
    type ActiveTask: ActiveTask<Backend = Self>;
}

#[async_trait]
pub trait ActiveTask: Send + Sync + Sized + 'static {
    type Backend: DownloadBackend<ActiveTask = Self>;

    async fn pause(self) -> Result<ResumeArtifactPath, String>;
    async fn cancel(self);
}

#[derive(Clone, Debug)]
struct MockBackend;

#[derive(Clone, Debug)]
struct MockActiveTask {
    pause_result: Result<ResumeArtifactPath, String>,
}

impl DownloadBackend for MockBackend {
    type ActiveTask = MockActiveTask;
}

#[async_trait]
impl ActiveTask for MockActiveTask {
    type Backend = MockBackend;

    async fn pause(self) -> Result<ResumeArtifactPath, String> {
        self.pause_result
    }

    async fn cancel(self) {}
}

#[derive(Debug, PartialEq, Eq)]
enum DownloadActorEffect {
    ReplyOk,
    ReplyError(String),
    SetStickyError(String),
    UpdateProgress {
        downloaded_bytes: u64,
        total_bytes: u64,
    },
    CompleteWaitersWithError(String),
}

#[derive(Default)]
struct DispatchContext {
    effects: Vec<DownloadActorEffect>,
}

#[allow(dead_code)]
#[derive(Debug, PartialEq, Eq)]
pub enum FsmEvent {
    Download,
    Pause,
    Cancel,
    ProgressUpdate {
        generation: ActiveDownloadGeneration,
        downloaded_bytes: u64,
        total_bytes: u64,
    },
    BackendError {
        generation: ActiveDownloadGeneration,
        message: String,
    },
}

#[derive(Debug)]
struct DownloadFsm<B: DownloadBackend> {
    _backend: PhantomData<B>,
}

impl<B: DownloadBackend> Default for DownloadFsm<B> {
    fn default() -> Self {
        Self {
            _backend: PhantomData,
        }
    }
}

#[state_machine(
    initial = "State::<B>::not_downloaded()",
    state(derive(Debug, PartialEq, Eq)),
    superstate(derive(Debug, PartialEq, Eq))
)]
impl<B: DownloadBackend> DownloadFsm<B> {
    #[superstate]
    async fn idle(
        event: &FsmEvent,
        context: &mut DispatchContext,
    ) -> Outcome<State<B>> {
        match event {
            FsmEvent::Pause => {
                context.effects.push(DownloadActorEffect::ReplyError("invalid transition".to_string()));
                Handled
            },
            _ => Super,
        }
    }

    #[state(superstate = "idle")]
    async fn not_downloaded(event: &FsmEvent) -> Outcome<State<B>> {
        match event {
            FsmEvent::Download => Handled,
            _ => Super,
        }
    }

    #[state(superstate = "idle")]
    async fn downloaded(event: &FsmEvent) -> Outcome<State<B>> {
        match event {
            FsmEvent::Download => Handled,
            _ => Super,
        }
    }

    #[state]
    async fn paused(
        resume_artifact_path: &mut ResumeArtifactPath,
        event: &FsmEvent,
        context: &mut DispatchContext,
    ) -> Outcome<State<B>> {
        match event {
            FsmEvent::Download => {
                let _ = resume_artifact_path;
                Handled
            },
            FsmEvent::Pause => {
                context.effects.push(DownloadActorEffect::ReplyOk);
                Handled
            },
            _ => Super,
        }
    }

    #[state]
    async fn downloading(
        active_task: &mut Option<B::ActiveTask>,
        generation: &mut ActiveDownloadGeneration,
        event: &FsmEvent,
        context: &mut DispatchContext,
    ) -> Outcome<State<B>> {
        match event {
            FsmEvent::ProgressUpdate {
                generation: event_generation,
                downloaded_bytes,
                total_bytes,
            } if event_generation == generation => {
                context.effects.push(DownloadActorEffect::UpdateProgress {
                    downloaded_bytes: *downloaded_bytes,
                    total_bytes: *total_bytes,
                });
                Handled
            },
            FsmEvent::BackendError {
                generation: event_generation,
                message,
            } if event_generation == generation => {
                if let Some(active_task) = active_task.take() {
                    active_task.cancel().await;
                }
                context.effects.push(DownloadActorEffect::SetStickyError(message.clone()));
                context.effects.push(DownloadActorEffect::CompleteWaitersWithError(message.clone()));
                Transition(State::<B>::not_downloaded())
            },
            FsmEvent::Pause => match active_task.take() {
                Some(active_task) => match active_task.pause().await {
                    Ok(resume_artifact_path) => {
                        context.effects.push(DownloadActorEffect::ReplyOk);
                        Transition(State::<B>::paused(resume_artifact_path))
                    },
                    Err(message) => {
                        context.effects.push(DownloadActorEffect::SetStickyError(message.clone()));
                        Transition(State::<B>::not_downloaded())
                    },
                },
                None => {
                    context.effects.push(DownloadActorEffect::SetStickyError("missing active task".to_string()));
                    Transition(State::<B>::not_downloaded())
                },
            },
            FsmEvent::Cancel => {
                if let Some(active_task) = active_task.take() {
                    active_task.cancel().await;
                }
                Transition(State::<B>::not_downloaded())
            },
            FsmEvent::ProgressUpdate {
                ..
            }
            | FsmEvent::BackendError {
                ..
            } => Handled,
            _ => Super,
        }
    }
}

#[tokio::test]
async fn test_phase0_statig_macro_shape_supports_consumable_active_task() {
    let active_task = MockActiveTask {
        pause_result: Ok(ResumeArtifactPath(PathBuf::from("resume.data"))),
    };
    let mut state_machine = DownloadFsm::<MockBackend>::default()
        .uninitialized_state_machine()
        .init_with_context(&mut DispatchContext::default())
        .await;

    unsafe {
        *state_machine.state_mut() = State::<MockBackend>::downloading(Some(active_task), ActiveDownloadGeneration(7));
    }

    let mut dispatch_context = DispatchContext::default();
    state_machine.handle_with_context(&FsmEvent::Pause, &mut dispatch_context).await;

    assert_eq!(dispatch_context.effects, [DownloadActorEffect::ReplyOk]);
    assert!(matches!(state_machine.state(), State::Paused { .. }));
}

#[tokio::test]
async fn test_phase0_statig_macro_shape_rejects_stale_backend_generation() {
    let active_task = MockActiveTask {
        pause_result: Ok(ResumeArtifactPath(PathBuf::from("resume.data"))),
    };
    let mut state_machine = DownloadFsm::<MockBackend>::default()
        .uninitialized_state_machine()
        .init_with_context(&mut DispatchContext::default())
        .await;

    unsafe {
        *state_machine.state_mut() = State::<MockBackend>::downloading(Some(active_task), ActiveDownloadGeneration(7));
    }

    let mut dispatch_context = DispatchContext::default();
    state_machine
        .handle_with_context(
            &FsmEvent::ProgressUpdate {
                generation: ActiveDownloadGeneration(6),
                downloaded_bytes: 100,
                total_bytes: 200,
            },
            &mut dispatch_context,
        )
        .await;

    assert!(dispatch_context.effects.is_empty());
    assert!(matches!(state_machine.state(), State::Downloading { .. }));
}

struct NoMacroDownloadFsm<B: DownloadBackend> {
    _backend: PhantomData<B>,
}

impl<B: DownloadBackend> Default for NoMacroDownloadFsm<B> {
    fn default() -> Self {
        Self {
            _backend: PhantomData,
        }
    }
}

enum NoMacroState<B: DownloadBackend> {
    NotDownloaded,
    Downloading {
        active_task: Option<B::ActiveTask>,
        generation: ActiveDownloadGeneration,
    },
}

impl<B: DownloadBackend> statig::awaitable::IntoStateMachine for NoMacroDownloadFsm<B> {
    type Context<'context> = DispatchContext;
    type Event<'event> = FsmEvent;
    type State = NoMacroState<B>;
    type Superstate<'superstate> = ();

    fn initial() -> Self::State {
        NoMacroState::NotDownloaded
    }
}

impl<B: DownloadBackend> statig::awaitable::State<NoMacroDownloadFsm<B>> for NoMacroState<B> {
    fn call_handler(
        &mut self,
        _shared_storage: &mut NoMacroDownloadFsm<B>,
        event: &<NoMacroDownloadFsm<B> as statig::awaitable::IntoStateMachine>::Event<'_>,
        context: &mut <NoMacroDownloadFsm<B> as statig::awaitable::IntoStateMachine>::Context<'_>,
    ) -> impl Future<Output = Outcome<Self>> {
        async move {
            match self {
                NoMacroState::NotDownloaded => match event {
                    FsmEvent::Download => Handled,
                    _ => Super,
                },
                NoMacroState::Downloading {
                    active_task,
                    generation,
                } => match event {
                    FsmEvent::ProgressUpdate {
                        generation: event_generation,
                        downloaded_bytes,
                        total_bytes,
                    } if event_generation == generation => {
                        context.effects.push(DownloadActorEffect::UpdateProgress {
                            downloaded_bytes: *downloaded_bytes,
                            total_bytes: *total_bytes,
                        });
                        Handled
                    },
                    FsmEvent::Cancel => {
                        if let Some(active_task) = active_task.take() {
                            active_task.cancel().await;
                        }
                        Transition(NoMacroState::NotDownloaded)
                    },
                    FsmEvent::ProgressUpdate {
                        ..
                    }
                    | FsmEvent::BackendError {
                        ..
                    } => Handled,
                    _ => Super,
                },
            }
        }
    }
}

#[tokio::test]
async fn test_phase0_statig_no_macro_shape_supports_same_core_contracts() {
    let active_task = MockActiveTask {
        pause_result: Ok(ResumeArtifactPath(PathBuf::from("resume.data"))),
    };
    let mut state_machine = NoMacroDownloadFsm::<MockBackend>::default()
        .uninitialized_state_machine()
        .init_with_context(&mut DispatchContext::default())
        .await;

    unsafe {
        *state_machine.state_mut() = NoMacroState::Downloading {
            active_task: Some(active_task),
            generation: ActiveDownloadGeneration(11),
        };
    }

    let mut dispatch_context = DispatchContext::default();
    state_machine
        .handle_with_context(
            &FsmEvent::ProgressUpdate {
                generation: ActiveDownloadGeneration(11),
                downloaded_bytes: 10,
                total_bytes: 20,
            },
            &mut dispatch_context,
        )
        .await;

    assert_eq!(
        dispatch_context.effects,
        [DownloadActorEffect::UpdateProgress {
            downloaded_bytes: 10,
            total_bytes: 20,
        }]
    );
}
