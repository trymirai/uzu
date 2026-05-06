use std::any::Any;

use iocraft::prelude::*;
use nagare::text_to_speech::{TextToSpeechSession, TextToSpeechSessionStreamChunk};
use shoji::types::{basic::CancelToken, model::Model, session::text_to_speech::TextToSpeechStats};

use crate::{
    cli::{
        components::{ApplicationState, HistoryCellType},
        helpers::HINT_SESSION_INTERRUPT,
        sessions::SessionState,
    },
    player::Player,
};

#[derive(Clone)]
pub struct TextToSpeechSessionRuntime {
    pub session: TextToSpeechSession,
    pub player: Player,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TextToSpeechSessionStatus {
    Idle,
    Loading,
    Synthesizing,
}

#[derive(Clone)]
pub struct TextToSpeechSessionState {
    runtime: Option<TextToSpeechSessionRuntime>,
    pending_stats: Option<TextToSpeechStats>,
    cancel_token: Option<CancelToken>,
    status: TextToSpeechSessionStatus,
}

impl TextToSpeechSessionState {
    pub fn loading() -> Self {
        Self {
            runtime: None,
            pending_stats: None,
            cancel_token: None,
            status: TextToSpeechSessionStatus::Loading,
        }
    }

    pub fn idle(runtime: TextToSpeechSessionRuntime) -> Self {
        Self {
            runtime: Some(runtime),
            pending_stats: None,
            cancel_token: None,
            status: TextToSpeechSessionStatus::Idle,
        }
    }
}

impl SessionState for TextToSpeechSessionState {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn is_busy(&self) -> bool {
        matches!(self.status, TextToSpeechSessionStatus::Loading | TextToSpeechSessionStatus::Synthesizing)
    }

    fn interrupt(&self) -> bool {
        let Some(cancel_token) = self.cancel_token.as_ref() else {
            return false;
        };
        cancel_token.cancel();
        true
    }

    fn status_text(&self) -> Option<String> {
        let status = match self.status {
            TextToSpeechSessionStatus::Idle => "loaded".to_string(),
            TextToSpeechSessionStatus::Loading => "loading".to_string(),
            TextToSpeechSessionStatus::Synthesizing => format!("synthesizing ({})", HINT_SESSION_INTERRUPT),
        };
        Some(status)
    }

    fn pending_history_cell(&self) -> Option<HistoryCellType> {
        self.pending_stats.clone().map(|stats| HistoryCellType::TextToSpeechOutput {
            stats,
        })
    }
}

pub async fn ensure_session(
    state: State<ApplicationState>,
    model: &Model,
) -> Option<TextToSpeechSessionRuntime> {
    let mut state = state;
    {
        let state = state.read();
        if let Some(runtime) =
            text_to_speech_state(&state).and_then(|text_to_speech_state| text_to_speech_state.runtime.clone())
        {
            return Some(runtime);
        }
    }

    {
        let mut state = state.write();
        if let Some(model_state) = state.model_state.as_mut() {
            model_state.session_state = Some(Box::new(TextToSpeechSessionState::loading()));
        }
    }

    let engine = state.read().engine.clone();
    let session = match engine.text_to_speech(model.clone()).await {
        Ok(session) => session,
        Err(error) => {
            let mut state = state.write();
            if let Some(model_state) = state.model_state.as_mut() {
                model_state.session_state = None;
            }
            state.history.push(HistoryCellType::CommandResult {
                result: format!("Failed to load session: {}", error),
            });
            return None;
        },
    };
    let player = match Player::new() {
        Ok(player) => player,
        Err(error) => {
            let mut state = state.write();
            if let Some(model_state) = state.model_state.as_mut() {
                model_state.session_state = None;
            }
            state.history.push(HistoryCellType::CommandResult {
                result: format!("Failed to initialize player: {}", error),
            });
            return None;
        },
    };

    let runtime = TextToSpeechSessionRuntime {
        session,
        player,
    };
    {
        let mut state = state.write();
        if let Some(model_state) = state.model_state.as_mut() {
            model_state.session_state = Some(Box::new(TextToSpeechSessionState::idle(runtime.clone())));
        }
    }
    Some(runtime)
}

pub async fn run_session(
    state: State<ApplicationState>,
    runtime: TextToSpeechSessionRuntime,
    text: String,
) {
    let mut state = state;
    {
        let mut state = state.write();
        if let Some(text_to_speech_state) = text_to_speech_state_mut(&mut state) {
            text_to_speech_state.runtime.as_ref().map(|runtime| runtime.player.stop());
            text_to_speech_state.pending_stats = None;
            text_to_speech_state.cancel_token = None;
            text_to_speech_state.status = TextToSpeechSessionStatus::Synthesizing;
        }
    }

    let stream = runtime.session.synthesize_stream(text).await;
    let cancel_token = stream.cancel_token();
    {
        let mut state = state.write();
        if let Some(text_to_speech_state) = text_to_speech_state_mut(&mut state) {
            text_to_speech_state.cancel_token = Some(cancel_token.clone());
        }
    }

    let mut latest_stats: Option<TextToSpeechStats> = None;
    let mut result_error: Option<String> = None;

    while let Some(chunk) = stream.next().await {
        match chunk {
            TextToSpeechSessionStreamChunk::Output {
                output,
            } => {
                let stats = output.stats.clone();
                if let Err(error) = runtime.player.append_pcm_batch(output.pcm_batch) {
                    cancel_token.cancel();
                    result_error = Some(format!("Playback error: {}", error));
                    break;
                }
                let mut state = state.write();
                if let Some(text_to_speech_state) = text_to_speech_state_mut(&mut state) {
                    text_to_speech_state.pending_stats = Some(stats.clone());
                }
                latest_stats = Some(stats);
            },
            TextToSpeechSessionStreamChunk::Error {
                error,
            } => {
                result_error = Some(format!("Text to speech error: {}", error));
                break;
            },
        }
    }

    let mut state = state.write();
    let final_stats = if let Some(text_to_speech_state) = text_to_speech_state_mut(&mut state) {
        let final_stats = latest_stats.or_else(|| text_to_speech_state.pending_stats.take());
        text_to_speech_state.pending_stats = None;
        text_to_speech_state.cancel_token = None;
        text_to_speech_state.status = TextToSpeechSessionStatus::Idle;
        final_stats
    } else {
        latest_stats
    };

    if let Some(result) = result_error {
        state.history.push(HistoryCellType::CommandResult {
            result,
        });
    } else if let Some(stats) = final_stats {
        state.history.push(HistoryCellType::TextToSpeechOutput {
            stats,
        });
    } else if cancel_token.is_cancelled() {
        state.history.push(HistoryCellType::CommandResult {
            result: "Text to speech cancelled".to_string(),
        });
    } else {
        state.history.push(HistoryCellType::CommandResult {
            result: "Text to speech error: no response".to_string(),
        });
    }
}

fn text_to_speech_state(state: &ApplicationState) -> Option<&TextToSpeechSessionState> {
    state
        .model_state
        .as_ref()
        .and_then(|model_state| model_state.session_state.as_deref())
        .and_then(|session_state| session_state.as_any().downcast_ref::<TextToSpeechSessionState>())
}

fn text_to_speech_state_mut(state: &mut ApplicationState) -> Option<&mut TextToSpeechSessionState> {
    state
        .model_state
        .as_mut()
        .and_then(|model_state| model_state.session_state.as_deref_mut())
        .and_then(|session_state| session_state.as_any_mut().downcast_mut::<TextToSpeechSessionState>())
}
