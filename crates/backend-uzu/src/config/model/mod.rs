use proc_macros::uzu_config_abstract;

use crate::config::token_codec::AnyTokenCodecConfig;

pub mod classifier_model;
pub mod generation;
pub mod language_model;

#[uzu_config_abstract(language_model::LanguageModelConfig, classifier_model::ClassifierModelConfig)]
pub struct ModelConfig {
    pub token_codec_config: AnyTokenCodecConfig,
}
