use std::any::Any;

use iocraft::prelude::*;
use nagare::decision::{DecisionSession, DecisionSessionError};
use shoji::types::{
    model::Model,
    session::decision::{DecisionInput, DecisionOption, DecisionQuestion},
};

use crate::interactive::{
    components::{ApplicationState, HistoryCellType},
    sessions::SessionState,
};

const SCORE_LEVELS_SEPARATOR: char = '<';
const OPTION_DESCRIPTION_SEPARATOR: char = ':';

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecisionSessionStatus {
    Idle,
    Loading,
    Deciding,
}

// Without a current state the session waits for one (state mode); with it, every input is a question about it
// (question mode).
#[derive(Clone)]
pub struct DecisionSessionState {
    session: Option<DecisionSession>,
    status: DecisionSessionStatus,
    current_state: Option<String>,
}

impl DecisionSessionState {
    pub fn loading() -> Self {
        Self {
            session: None,
            status: DecisionSessionStatus::Loading,
            current_state: None,
        }
    }

    pub fn idle(session: DecisionSession) -> Self {
        Self {
            session: Some(session),
            status: DecisionSessionStatus::Idle,
            current_state: None,
        }
    }
}

impl SessionState for DecisionSessionState {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn is_busy(&self) -> bool {
        matches!(self.status, DecisionSessionStatus::Loading | DecisionSessionStatus::Deciding)
    }

    fn interrupt(&self) -> bool {
        false
    }

    fn status_text(&self) -> Option<String> {
        let status = match self.status {
            DecisionSessionStatus::Idle => "loaded".to_string(),
            DecisionSessionStatus::Loading => "loading".to_string(),
            DecisionSessionStatus::Deciding => "deciding".to_string(),
        };
        Some(status)
    }

    fn pending_history_cell(&self) -> Option<HistoryCellType> {
        None
    }
}

pub async fn ensure_session(
    state: State<ApplicationState>,
    model: &Model,
) -> Option<DecisionSession> {
    let mut state = state;
    {
        let state = state.read();
        if let Some(session) = decision_state(&state).and_then(|decision_state| decision_state.session.clone()) {
            return Some(session);
        }
    }

    {
        let mut state = state.write();
        if let Some(model_state) = state.model_state.as_mut() {
            model_state.session_state = Some(Box::new(DecisionSessionState::loading()));
        }
    }

    let engine = state.read().engine.clone();
    let session = match engine.decision(model.clone()).await {
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

    {
        let mut state = state.write();
        if let Some(model_state) = state.model_state.as_mut() {
            model_state.session_state = Some(Box::new(DecisionSessionState::idle(session.clone())));
        }
    }
    Some(session)
}

// In state mode the input becomes the current state; in question mode it's a question answered for that state.
pub async fn run_session(
    state: State<ApplicationState>,
    session: DecisionSession,
    text: String,
) {
    let mut state = state;
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    let current_state = decision_state(&state.read()).and_then(|decision_state| decision_state.current_state.clone());
    let Some(current_state) = current_state else {
        let mut state = state.write();
        if let Some(decision_state) = decision_state_mut(&mut state) {
            decision_state.current_state = Some(text.to_string());
        }
        state.history.push(HistoryCellType::CommandResult {
            result: "Text set, now ask questions about it".to_string(),
        });
        return;
    };

    let question = parse_question(text);
    set_status(&mut state.write(), DecisionSessionStatus::Deciding);
    let input = DecisionInput {
        state: current_state,
        questions: vec![question.clone()],
    };
    let output = session.decide(input).await;
    let mut state = state.write();
    set_status(&mut state, DecisionSessionStatus::Idle);
    match output.map(|output| output.results.into_iter().next()) {
        Ok(Some(result)) => state.history.push(HistoryCellType::DecisionOutput {
            question,
            result,
        }),
        Ok(None) => {},
        Err(DecisionSessionError::InvalidQuestion {
            message,
            ..
        }) => state.history.push(HistoryCellType::CommandResult {
            result: format!("Invalid question: {message}"),
        }),
        Err(error) => state.history.push(HistoryCellType::CommandResult {
            result: format!("Decision error: {}", error),
        }),
    }
}

pub fn is_question_mode(state: &ApplicationState) -> bool {
    decision_state(state).is_some_and(|decision_state| decision_state.current_state.is_some())
}

// Esc in question mode goes back to state mode; otherwise it's left to the application.
pub fn leave_question_mode(state: &mut ApplicationState) -> bool {
    let Some(decision_state) = decision_state_mut(state) else {
        return false;
    };
    if decision_state.status != DecisionSessionStatus::Idle || decision_state.current_state.is_none() {
        return false;
    }
    decision_state.current_state = None;
    true
}

// The first line is the instructions. No more lines make a yes/no question, one line of levels
// joined by `<` a score question, and one `key` or `key: description` per line a choice question.
// Option counts and duplicates are left to the model's prompt template, which rejects bad questions.
fn parse_question(text: &str) -> DecisionQuestion {
    let mut lines = text.lines().map(str::trim);
    let instructions = lines.next().filter(|line| !line.is_empty()).map(String::from);
    let options = lines.filter(|line| !line.is_empty()).collect::<Vec<_>>();
    match options.as_slice() {
        [] => DecisionQuestion::YesNo {
            instructions,
            no_description: None,
            yes_description: None,
        },
        [levels] if levels.contains(SCORE_LEVELS_SEPARATOR) => DecisionQuestion::Score {
            instructions,
            levels: levels.split(SCORE_LEVELS_SEPARATOR).map(|level| level.trim().to_string()).collect(),
        },
        options => DecisionQuestion::Choice {
            instructions,
            options: options
                .iter()
                .map(|option| match option.split_once(OPTION_DESCRIPTION_SEPARATOR) {
                    Some((key, description)) => DecisionOption {
                        key: key.trim().to_string(),
                        description: Some(description.trim())
                            .filter(|description| !description.is_empty())
                            .map(String::from),
                    },
                    None => DecisionOption {
                        key: option.to_string(),
                        description: None,
                    },
                })
                .collect(),
        },
    }
}

pub fn question_title(question: &DecisionQuestion) -> String {
    let (instructions, fallback) = match question {
        DecisionQuestion::Choice {
            instructions,
            ..
        } => (instructions, "Choice"),
        DecisionQuestion::Score {
            instructions,
            ..
        } => (instructions, "Score"),
        DecisionQuestion::YesNo {
            instructions,
            ..
        } => (instructions, "Yes or no"),
    };
    instructions.clone().unwrap_or_else(|| fallback.to_string())
}

fn set_status(
    state: &mut ApplicationState,
    status: DecisionSessionStatus,
) {
    if let Some(decision_state) = decision_state_mut(state) {
        decision_state.status = status;
    }
}

fn decision_state(state: &ApplicationState) -> Option<&DecisionSessionState> {
    state
        .model_state
        .as_ref()
        .and_then(|model_state| model_state.session_state.as_deref())
        .and_then(|session_state| session_state.as_any().downcast_ref::<DecisionSessionState>())
}

fn decision_state_mut(state: &mut ApplicationState) -> Option<&mut DecisionSessionState> {
    state
        .model_state
        .as_mut()
        .and_then(|model_state| model_state.session_state.as_deref_mut())
        .and_then(|session_state| session_state.as_any_mut().downcast_mut::<DecisionSessionState>())
}
