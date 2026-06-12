use proc_macros::uzu_config;

use crate::config::classifier::ClassifierConfig;

#[uzu_config(super::ModelConfig)]
pub struct ClassifierModelConfig {
    pub classifier_config: ClassifierConfig,
}
