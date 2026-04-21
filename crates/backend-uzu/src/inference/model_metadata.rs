use std::{fs::File, io::BufReader, path::Path};

use shoji::types::model::Specialization;

use crate::config::{ModelMetadata as InnerModelMetadata, ModelType};

pub struct ModelMetadata {
    pub toolchain_version: String,
    pub specialization: Specialization,
}

pub fn resolve_model_metadata(model_path: &Path) -> Option<ModelMetadata> {
    let config_path = model_path.join("config.json");
    let file = File::open(&config_path).ok()?;
    let metadata: InnerModelMetadata = serde_json::from_reader(BufReader::new(file)).ok()?;
    let specialization = match metadata.model_type {
        ModelType::LanguageModel => Specialization::Chat,
        ModelType::ClassifierModel => Specialization::Classification,
        ModelType::TtsModel => Specialization::TextToSpeech,
    };
    Some(ModelMetadata {
        toolchain_version: metadata.toolchain_version,
        specialization,
    })
}
