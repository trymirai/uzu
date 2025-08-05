use std::{fs::File, io::BufReader, path::PathBuf};

use serde::de::DeserializeOwned;

use crate::session::config::{
    generation_config::GenerationConfig, tokenizer_config::TokenizerConfig,
};

#[derive(Clone, Debug)]
pub struct GenerationMetadata {
    pub generation_config: Option<GenerationConfig>,
    pub tokenizer_config: Option<TokenizerConfig>,
    pub chat_template: Option<String>,
}

impl GenerationMetadata {
    pub fn load(model_path: PathBuf) -> Self {
        let generation_config: Option<GenerationConfig> =
            Self::read_file_as_struct(
                &model_path.join("generation_config.json"),
            );

        let tokenizer_config: Option<TokenizerConfig> =
            Self::read_file_as_struct(
                &model_path.join("tokenizer_config.json"),
            );

        let chat_template: Option<String> =
            Self::read_file_as_string(&model_path.join("chat_template.jinja"));

        Self {
            generation_config,
            tokenizer_config,
            chat_template,
        }
    }

    fn read_file_as_struct<T: DeserializeOwned>(path: &PathBuf) -> Option<T> {
        if !path.exists() {
            return None;
        }

        let file = File::open(path);
        match file {
            Ok(file) => {
                let reader = BufReader::new(file);
                let result = serde_json::from_reader(reader)
                    .map_or(None, |value| Some(value));
                result
            },
            Err(_) => None,
        }
    }

    fn read_file_as_string(path: &PathBuf) -> Option<String> {
        if !path.exists() {
            return None;
        }

        std::fs::read_to_string(&path).map_or(None, |value| Some(value))
    }
}
