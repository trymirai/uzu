use uzu_engine_macros::uzu_config;

use crate::config::{decoder::DecoderConfig, model::generation::GenerationConfig};

#[uzu_config]
#[derive(Default)]
#[serde(rename_all = "snake_case")]
pub enum VocabularyOrder {
    #[default]
    Original,
    Frequency,
}

#[uzu_config(super::ModelConfig)]
pub struct LanguageModelConfig {
    pub decoder_config: DecoderConfig,
    pub generation_config: GenerationConfig,
    #[serde(default)]
    pub vocabulary_order: VocabularyOrder,
}

#[cfg(test)]
#[path = "../../../unit/config/language_model_test.rs"]
mod tests;
