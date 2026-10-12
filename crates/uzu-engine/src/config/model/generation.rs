use uzu_engine_macros::uzu_config;

use crate::{encodable_block::sampling::SamplingMethod, utils::strict_serde::Unsupported};

#[uzu_config]
pub struct GenerationConfig {
    pub stop_token_ids: Box<[u64]>,
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub banned_tokens: Option<Unsupported>,
    pub repetition_penalty: Option<f32>,
    pub presence_penalty: Option<Unsupported>,
    pub frequency_penalty: Option<Unsupported>,
    pub suffix_repetition_length: Option<u32>,
}

impl GenerationConfig {
    pub fn default_sampling_method(&self) -> SamplingMethod {
        SamplingMethod::Stochastic {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            repetition_penalty: self.repetition_penalty,
            suffix_repetition_length: self.suffix_repetition_length,
        }
    }
}
