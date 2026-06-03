use std::{fs::File, io::BufReader, path::Path};

use proc_macros::uzu_config;

use crate::{
    config::{decoder::DecoderConfig, model::generation::GenerationConfig},
    prelude::Error,
};

#[uzu_config(super::ModelConfig)]
pub struct LanguageModelConfig {
    pub decoder_config: DecoderConfig,
    pub generation_config: GenerationConfig,
}

impl LanguageModelConfig {
    pub fn new(model_path: &Path) -> Result<LanguageModelConfig, Error> {
        let config_path = model_path.join("config.json");
        let config_file = File::open(&config_path)?;
        let model_config: LanguageModelConfig = serde_json::from_reader(BufReader::new(config_file))?;
        Ok(model_config)
    }
}
