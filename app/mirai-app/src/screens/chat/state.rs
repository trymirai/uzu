//! `ChatState` — the chat's pure domain + UI data, separated from the GPUI
//! handles (entities, scroll handles, frame counters) that live on `ChatView`.
//! This is the TCA "State" principle: the data that intent methods mutate and
//! the view renders, holding no framework plumbing — so it can be reasoned
//! about (and tested) independently of a window.

use uzu::{
    session::chat::ChatSession,
    types::{basic::CancelToken, model::Model},
};

use super::{conversation::ChatMsg, sampling::SamplingMode};

pub(super) struct ChatState {
    pub messages: Vec<ChatMsg>,
    pub model: Option<Model>,
    pub streaming: bool,
    /// True between "send" and the first `Started` token — model is loading.
    pub waiting_for_model: bool,
    pub cancel: Option<CancelToken>,
    pub chat_id: Option<String>,
    pub created_at: u64,
    pub model_picker_open: bool,
    /// Message index whose per-message model menu is open.
    pub msg_model_picker_open: Option<usize>,
    /// Message index whose performance popover is open (`None` = closed).
    pub perf_open_msg: Option<usize>,
    pub file_upload_open: bool,
    /// Files attached to the next message: (display_name, extension, content).
    pub attached_files: Vec<(String, String, String)>,
    /// Name of the model the UI considers "loaded" (set once a chat runs).
    pub loaded_model: Option<String>,
    pub gen_settings_open: bool,
    pub sampling_mode: SamplingMode,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub min_p: f32,
    pub max_tokens: u32,
    pub chat_title: String,
    pub title_pending: bool,
    pub session: Option<ChatSession>,
    pub session_model_id: Option<String>,
}
