use std::any::Any;

use chrono::Local;
use iocraft::prelude::*;
use nagare::{
    chat::{ChatSession, ChatSessionStreamChunk},
    tool::uzu_tool_function,
};
use shoji::types::{
    basic::{CancelToken, ReasoningEffort, ToolCall},
    model::Model,
    session::chat::{ChatConfig, ChatMessage, ChatReplyConfig, ChatReplyStats, ChatRole},
};

use crate::interactive::{
    components::{ApplicationState, HistoryCellType, TranscriptItem},
    helpers::HINT_SESSION_INTERRUPT,
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
    pending_items: Vec<TranscriptItem>,
    pending_stats: Option<ChatReplyStats>,
    cancel_token: Option<CancelToken>,
    status: ChatSessionStatus,
}

impl ChatSessionState {
    pub fn loading() -> Self {
        Self {
            session: None,
            pending_items: Vec::new(),
            pending_stats: None,
            cancel_token: None,
            status: ChatSessionStatus::Loading,
        }
    }

    pub fn idle(session: ChatSession) -> Self {
        Self {
            session: Some(session),
            pending_items: Vec::new(),
            pending_stats: None,
            cancel_token: None,
            status: ChatSessionStatus::Idle,
        }
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
        matches!(self.status, ChatSessionStatus::Loading | ChatSessionStatus::Generating)
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
            ChatSessionStatus::Idle => "loaded".to_string(),
            ChatSessionStatus::Loading => "loading".to_string(),
            ChatSessionStatus::Generating => format!("generating ({})", HINT_SESSION_INTERRUPT),
        };
        Some(status)
    }

    fn pending_history_cell(&self) -> Option<HistoryCellType> {
        if self.pending_items.is_empty() && self.pending_stats.is_none() {
            return None;
        }
        Some(HistoryCellType::ChatTranscript {
            items: self.pending_items.clone(),
            stats: self.pending_stats.clone(),
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
    let session = match async {
        let mut session = engine.chat(model.clone(), ChatConfig::default()).await?;
        if session.supports_tool_calls().await {
            session.add_tool(get_current_date_time).await?;
            session.add_tool(sleep).await?;
        }
        Ok::<_, anyhow::Error>(session)
    }
    .await
    {
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
    let (thinking_support, thinking_override, thinking) = {
        let state = state.read();
        (
            state.model_state.as_ref().map(|model_state| model_state.capabilities.thinking).unwrap_or_default(),
            state.reasoning_effort_override(),
            state.thinking(),
        )
    };

    let reasoning_effort = match thinking_override {
        Some(effort) => match thinking_support.fulfill_requested_effort(effort) {
            Ok(effort) => effort,
            Err(error) => {
                state.write().history.push(HistoryCellType::CommandResult {
                    result: format!("Invalid --reasoning-effort: {error}"),
                });
                return;
            },
        },
        None => thinking_support.with_preference(&thinking).reasoning_effort(),
    };

    let mut messages = Vec::<ChatMessage>::new();
    if let Some(reasoning_effort) = reasoning_effort
        && session.messages().await.is_empty()
    {
        let system_message = ChatMessage::system().with_reasoning_effort(reasoning_effort);
        messages.push(system_message);
    };
    messages.push(ChatMessage::user().with_text(text));

    let reply_config = ChatReplyConfig::default().with_sampling_policy(state.read().preferences().sampling.policy());
    {
        let mut state = state.write();
        if let Some(chat_state) = chat_state_mut(&mut state) {
            chat_state.pending_items = Vec::new();
            chat_state.pending_stats = None;
            chat_state.cancel_token = None;
            chat_state.status = ChatSessionStatus::Generating;
        }
    }

    let history_offset = session.messages().await.len();
    let stream = session.reply_with_stream(messages, reply_config).await;
    let mut latest_stats: Option<ChatReplyStats> = None;
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
                if let Some(reply) = replies.last() {
                    latest_stats = Some(reply.stats.clone());
                }
                let items = build_transcript(&session.messages().await, history_offset);
                let mut state = state.write();
                if let Some(chat_state) = chat_state_mut(&mut state) {
                    chat_state.pending_items = items;
                    chat_state.pending_stats = latest_stats.clone();
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

    let final_items = build_transcript(&session.messages().await, history_offset);
    let mut state = state.write();
    if let Some(chat_state) = chat_state_mut(&mut state) {
        chat_state.pending_items = Vec::new();
        chat_state.pending_stats = None;
        chat_state.cancel_token = None;
        chat_state.status = ChatSessionStatus::Idle;
    }
    if !final_items.is_empty() || latest_stats.is_some() {
        state.history.push(HistoryCellType::ChatTranscript {
            items: final_items,
            stats: latest_stats,
        });
    }
}

/// A tool call is `calling` until its result message shows up in the history,
/// then it becomes `called`.
fn build_transcript(
    messages: &[ChatMessage],
    history_offset: usize,
) -> Vec<TranscriptItem> {
    let messages = &messages[history_offset.min(messages.len())..];
    let mut items = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        if message.role != (ChatRole::Assistant {}) {
            continue;
        }
        if let Some(reasoning) =
            message.reasoning().map(|content| content.trim().to_string()).filter(|content| !content.is_empty())
        {
            items.push(TranscriptItem::Thinking(reasoning));
        }
        if let Some(text) = message.text().filter(|content| !content.is_empty()) {
            items.push(TranscriptItem::Text(text));
        }
        for tool_call in message.tool_calls() {
            items.push(TranscriptItem::ToolCall {
                name: tool_call.name.clone(),
                called: tool_call_called(messages, index, &tool_call),
            });
        }
    }
    items
}

fn tool_call_called(
    messages: &[ChatMessage],
    index: usize,
    tool_call: &ToolCall,
) -> bool {
    messages[index + 1..].iter().any(|message| {
        message.role == (ChatRole::Tool {})
            && message.tool_call_results().iter().any(|(identifier, _, _)| match (&tool_call.identifier, identifier) {
                (Some(call_identifier), Some(result_identifier)) => call_identifier == result_identifier,
                _ => true,
            })
    })
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

/// Returns current date and time in RFC 3339 format: YYYY-MM-DDTHH:MM:SSZ
#[uzu_tool_function]
fn get_current_date_time() -> String {
    Local::now().to_rfc3339()
}

/// Sleeps for the given number of seconds before responding, at most 60 seconds.
#[uzu_tool_function]
async fn sleep(
    /// Number of seconds to sleep, at most 60.
    seconds: f64
) -> Result<String, String> {
    if !(0.0..=60.0).contains(&seconds) {
        return Err(format!("Refusing to sleep for {} seconds, must be between 0 and 60", seconds));
    }
    tokio::time::sleep(std::time::Duration::from_secs_f64(seconds)).await;
    Ok(format!("Slept for {} seconds", seconds))
}
