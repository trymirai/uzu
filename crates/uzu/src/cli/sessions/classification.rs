use std::any::Any;

use iocraft::prelude::*;
use nagare::classification::ClassificationSession;
use shoji::types::{model::Model, session::classification::ClassificationMessage};

use crate::cli::{
    components::{ApplicationState, HistoryCellType},
    sessions::SessionState,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ClassificationSessionStatus {
    Idle,
    Loading,
    Classifying,
}

#[derive(Clone)]
pub struct ClassificationSessionState {
    session: Option<ClassificationSession>,
    status: ClassificationSessionStatus,
}

impl ClassificationSessionState {
    pub fn loading() -> Self {
        Self {
            session: None,
            status: ClassificationSessionStatus::Loading,
        }
    }

    pub fn idle(session: ClassificationSession) -> Self {
        Self {
            session: Some(session),
            status: ClassificationSessionStatus::Idle,
        }
    }
}

impl SessionState for ClassificationSessionState {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn is_busy(&self) -> bool {
        matches!(self.status, ClassificationSessionStatus::Loading | ClassificationSessionStatus::Classifying)
    }

    fn interrupt(&self) -> bool {
        false
    }

    fn status_text(&self) -> Option<String> {
        let status = match self.status {
            ClassificationSessionStatus::Idle => "loaded",
            ClassificationSessionStatus::Loading => "loading",
            ClassificationSessionStatus::Classifying => "classifying",
        };
        Some(status.to_string())
    }

    fn pending_history_cell(&self) -> Option<HistoryCellType> {
        None
    }
}

pub async fn ensure_session(
    state: State<ApplicationState>,
    model: &Model,
) -> Option<ClassificationSession> {
    let mut state = state;
    {
        let state = state.read();
        if let Some(session) =
            classification_state(&state).and_then(|classification_state| classification_state.session.clone())
        {
            return Some(session);
        }
    }

    {
        let mut state = state.write();
        if let Some(model_state) = state.model_state.as_mut() {
            model_state.session_state = Some(Box::new(ClassificationSessionState::loading()));
        }
    }

    let engine = state.read().engine.clone();
    let session = match engine.classification(model.clone()).await {
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
            model_state.session_state = Some(Box::new(ClassificationSessionState::idle(session.clone())));
        }
    }
    Some(session)
}

pub async fn run_session(
    state: State<ApplicationState>,
    session: ClassificationSession,
    text: String,
) {
    let mut state = state;
    {
        let mut state = state.write();
        if let Some(classification_state) = classification_state_mut(&mut state) {
            classification_state.status = ClassificationSessionStatus::Classifying;
        }
    }

    let message = ClassificationMessage::user(text);
    let output = session.classify(vec![message]).await;

    let mut state = state.write();
    if let Some(classification_state) = classification_state_mut(&mut state) {
        classification_state.status = ClassificationSessionStatus::Idle;
    }
    match output {
        Ok(output) => state.history.push(HistoryCellType::ClassificationOutput {
            output,
        }),
        Err(error) => state.history.push(HistoryCellType::CommandResult {
            result: format!("Classification error: {}", error),
        }),
    }
}

fn classification_state(state: &ApplicationState) -> Option<&ClassificationSessionState> {
    state
        .model_state
        .as_ref()
        .and_then(|model_state| model_state.session_state.as_deref())
        .and_then(|session_state| session_state.as_any().downcast_ref::<ClassificationSessionState>())
}

fn classification_state_mut(state: &mut ApplicationState) -> Option<&mut ClassificationSessionState> {
    state
        .model_state
        .as_mut()
        .and_then(|model_state| model_state.session_state.as_deref_mut())
        .and_then(|session_state| session_state.as_any_mut().downcast_mut::<ClassificationSessionState>())
}
