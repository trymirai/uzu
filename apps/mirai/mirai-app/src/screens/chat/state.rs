use uzu::{
    session::chat::ChatSession,
    types::{basic::CancelToken, model::Model},
};

use super::{chat_turn::ChatTurn, sampling::SamplingMode};
use crate::components::markdown::ParsedMarkdown;

pub(super) struct ChatState {
    pub messages: Vec<ChatTurn>,
    pub model: Option<Model>,
    pub streaming: bool,
    pub waiting_for_model: bool,
    pub cancel: Option<CancelToken>,
    pub stream_gen: u64,
    pub revealed_chars: usize,
    pub stream_parsed: Option<ParsedMarkdown>,
    pub stream_stable_len: usize,
    pub stream_parse_in_flight: bool,
    pub stream_parse_pending: bool,
    pub chat_id: Option<String>,
    pub created_at: u64,
    pub model_picker_open: bool,
    pub msg_model_picker_open: Option<usize>,
    pub pending_regen: Option<usize>,
    pub perf_open_msg: Option<usize>,
    pub file_upload_open: bool,
    pub attached_files: Vec<(String, String, String)>,
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
