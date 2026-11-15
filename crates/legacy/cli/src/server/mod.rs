mod error;
mod log;
mod model_admin;
mod model_slot;
mod loader;
mod request_info;

pub mod access_log;
pub mod audio;
pub mod chat_completions;
pub mod chat_tool_calls;
pub mod completions_raw;
pub mod decisions;
pub mod models;
pub mod request_log;
pub mod runner;
pub mod state;
pub mod vlm_reply;
pub mod vlm_session;

pub use audio::{
    handle_audio_capabilities, handle_audio_speech, handle_audio_transcriptions_json,
    handle_audio_transcriptions_multipart, handle_audio_translate, handle_live_asr_chunk, handle_live_asr_start,
    handle_live_asr_stop, handle_omni_capabilities, handle_web_ui,
};
pub use chat_completions::{ChatState, handle_chat_completions};
pub use completions_raw::handle_raw_completions;
pub use decisions::{
    DecisionBackend, DecisionState, handle_decision_model_load, handle_decision_model_unload, handle_decisions,
    handle_system_one,
};
pub use model_admin::{handle_model_load, handle_model_status, handle_model_unload};
pub use model_slot::{SlotSnapshot, SwapSlot};
pub use models::{handle_models, handle_models_static};
pub use runner::run_server;
pub use state::{ServerState, handle_server_state};
