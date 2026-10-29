use uzu_engine_macros::uzu_config;

use crate::{config::decoder::DecoderConfig, engine::language_model::GenerationConfig};

#[uzu_config(super::ModelConfig)]
pub struct DecisionModelConfig {
    pub decoder_config: DecoderConfig,
    pub generation_config: GenerationConfig,
    pub decision: DecisionConfig,
}

#[uzu_config]
pub struct DecisionConfig {
    pub codes: Box<[String]>,
    pub temperature: f32,
}
