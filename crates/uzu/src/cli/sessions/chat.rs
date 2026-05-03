use std::any::Any;

use iocraft::prelude::*;
use nagare::chat::{ChatSession, ChatSessionStreamChunk};
use shoji::types::{
    basic::CancelToken,
    model::Model,
    session::chat::{ChatConfig, ChatMessage, ChatReply, ChatReplyConfig},
};

use crate::cli::{
    components::{ApplicationState, HistoryCellType},
    sessions::SessionState,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ChatSessionStatus {
    Idle,
    Loading,
    Generating,
}

#[derive(Clone)]
pub struct ChatSessionState {
    session: Option<ChatSession>,
    pending_reply: Option<ChatReply>,
    cancel_token: Option<CancelToken>,
    status: ChatSessionStatus,
}

impl ChatSessionState {
    pub fn loading() -> Self {
        Self {
            session: None,
            pending_reply: None,
            cancel_token: None,
            status: ChatSessionStatus::Loading,
        }
    }

    pub fn idle(session: ChatSession) -> Self {
        Self {
            session: Some(session),
            pending_reply: None,
            cancel_token: None,
            status: ChatSessionStatus::Idle,
        }
    }

    fn is_busy(&self) -> bool {
        matches!(self.status, ChatSessionStatus::Loading | ChatSessionStatus::Generating)
    }
}

impl SessionState for ChatSessionState {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn is_busy(&self) -> bool {
        self.is_busy()
    }

    fn cancel(&self) -> bool {
        let Some(cancel_token) = self.cancel_token.as_ref() else {
            return false;
        };
        cancel_token.cancel();
        true
    }

    fn status_text(&self) -> Option<String> {
        let status = match self.status {
            ChatSessionStatus::Idle => "loaded",
            ChatSessionStatus::Loading => "loading...",
            ChatSessionStatus::Generating => "generating...",
        };
        Some(status.to_string())
    }

    fn pending_history_cell(&self) -> Option<HistoryCellType> {
        self.pending_reply.clone().map(|reply| HistoryCellType::ChatReply {
            reply,
        })
    }
}

pub async fn ensure_session(
    state: State<ApplicationState>,
    model: &Model,
) -> Option<ChatSession> {
    let mut state = state;
    {
        let state = state.read();
        if let Some(session) = chat_state(&state).and_then(|chat_state| chat_state.session.clone()) {
            return Some(session);
        }
    }

    {
        let mut state = state.write();
        if let Some(model_state) = state.model_state.as_mut() {
            model_state.session_state = Some(Box::new(ChatSessionState::loading()));
        }
    }

    let engine = state.read().engine.clone();
    let session = match engine.chat(model.clone(), ChatConfig::default()).await {
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
            model_state.session_state = Some(Box::new(ChatSessionState::idle(session.clone())));
        }
    }
    Some(session)
}

pub async fn run_session(
    state: State<ApplicationState>,
    session: ChatSession,
    text: String,
) {
    let mut state = state;
    let user_message = ChatMessage::user().with_text(text);
    {
        let mut state = state.write();
        if let Some(chat_state) = chat_state_mut(&mut state) {
            chat_state.pending_reply = None;
            chat_state.cancel_token = None;
            chat_state.status = ChatSessionStatus::Generating;
        }
    }

    let stream = session.reply_with_stream(vec![user_message], ChatReplyConfig::default()).await;
    let mut latest_reply: Option<ChatReply> = None;
    {
        let mut state = state.write();
        if let Some(chat_state) = chat_state_mut(&mut state) {
            chat_state.cancel_token = Some(stream.cancel_token());
        }
    }

    while let Some(chunk) = stream.next().await {
        match chunk {
            ChatSessionStreamChunk::Replies {
                replies,
            } => {
                if let Some(reply) = replies.last().cloned() {
                    latest_reply = Some(reply.clone());
                    let mut state = state.write();
                    if let Some(chat_state) = chat_state_mut(&mut state) {
                        chat_state.pending_reply = Some(reply);
                    }
                }
            },
            ChatSessionStreamChunk::Error {
                error,
            } => {
                state.write().history.push(HistoryCellType::CommandResult {
                    result: format!("Chat error: {}", error),
                });
                break;
            },
        }
    }

    let mut state = state.write();
    let final_reply = if let Some(chat_state) = chat_state_mut(&mut state) {
        let final_reply = latest_reply.or_else(|| chat_state.pending_reply.take());
        chat_state.pending_reply = None;
        chat_state.cancel_token = None;
        chat_state.status = ChatSessionStatus::Idle;
        final_reply
    } else {
        latest_reply
    };
    if let Some(reply) = final_reply {
        state.history.push(HistoryCellType::ChatReply {
            reply,
        });
    }
}

fn chat_state(state: &ApplicationState) -> Option<&ChatSessionState> {
    state
        .model_state
        .as_ref()
        .and_then(|model_state| model_state.session_state.as_deref())
        .and_then(|session_state| session_state.as_any().downcast_ref::<ChatSessionState>())
}

fn chat_state_mut(state: &mut ApplicationState) -> Option<&mut ChatSessionState> {
    state
        .model_state
        .as_mut()
        .and_then(|model_state| model_state.session_state.as_deref_mut())
        .and_then(|session_state| session_state.as_any_mut().downcast_mut::<ChatSessionState>())
}
