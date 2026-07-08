use proc_macros::uzu_config_abstract;

use crate::config::token_codec::AnyTokenCodecConfig;

pub mod classifier_model;
pub mod dflash_speculator;
pub mod generation;
pub mod language_model;
pub mod weaver_speculator;

#[uzu_config_abstract(
    language_model::LanguageModelConfig,
    classifier_model::ClassifierModelConfig,
    dflash_speculator::DFlashSpeculatorConfig,
    weaver_speculator::WeaverSpeculatorConfig
)]
pub struct ModelConfig {
    pub token_codec_config: AnyTokenCodecConfig,
}
