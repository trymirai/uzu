use std::{fs::File, io::BufReader, path::Path};

use shoji::types::model::ModelSpecialization;

use crate::config::model::AnyModelConfig;

#[derive(Debug, thiserror::Error)]
pub enum ModelMetadataError {
    #[error("Unable to open model configuration: {0}")]
    UnableToOpenConfig(#[from] std::io::Error),
    #[error("Unable to deserialize model configuration: {0}")]
    UnableToDeserializeConfig(#[from] serde_json::Error),
}

pub fn resolve_model_specialization(model_path: &Path) -> Result<ModelSpecialization, ModelMetadataError> {
    let config_path = model_path.join("config.json");
    let file = File::open(&config_path)?;
    let config: AnyModelConfig = serde_json::from_reader(BufReader::new(file))?;
    Ok(match config {
        AnyModelConfig::LanguageModelConfig(_) => ModelSpecialization::Chat {},
        AnyModelConfig::ClassifierModelConfig(_) => ModelSpecialization::Classification {},
        AnyModelConfig::TTSModelConfig(_) => ModelSpecialization::TextToSpeech {},
    })
}
