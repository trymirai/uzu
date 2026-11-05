use uzu_engine_macros::uzu_config;

use crate::config::{decoder::DecoderConfig, model::generation::GenerationConfig};

#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum VocabularyOrder {
    Original,
    Frequency,
}

#[uzu_config(super::ModelConfig)]
pub struct LanguageModelConfig {
    pub decoder_config: DecoderConfig,
    pub generation_config: GenerationConfig,
    pub vocabulary_order: VocabularyOrder,
}
