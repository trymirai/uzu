use proc_macros::uzu_config;

use crate::utils::strict_serde::Unsupported;

#[uzu_config]
pub struct GenerationConfig {
    pub stop_token_ids: Box<[u64]>,
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub banned_tokens: Option<Unsupported>,
    pub repetition_penalty: Option<Unsupported>,
    pub presence_penalty: Option<Unsupported>,
    pub frequency_penalty: Option<Unsupported>,
    pub suffix_repetition_length: Option<Unsupported>,
}
