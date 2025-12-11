use serde::{Deserialize, Serialize};

use crate::{ClassifierModelConfig, LanguageModelConfig};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(untagged)]
pub enum ModelConfig {
    LanguageModel(LanguageModelConfig),
    Classifier(ClassifierModelConfig),
}

impl ModelConfig {
    pub fn as_language_model(&self) -> Option<&LanguageModelConfig> {
        match self {
            ModelConfig::LanguageModel(config) => Some(config),
            _ => None,
        }
    }

    pub fn as_classifier(&self) -> Option<&ClassifierModelConfig> {
        match self {
            ModelConfig::Classifier(config) => Some(config),
            _ => None,
        }
    }

    pub fn is_language_model(&self) -> bool {
        matches!(self, ModelConfig::LanguageModel(_))
    }

    pub fn is_classifier(&self) -> bool {
        matches!(self, ModelConfig::Classifier(_))
    }
}
