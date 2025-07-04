use std::{fs::File, io::BufReader, path::PathBuf};

use serde::Deserialize;

use crate::session::session_error::SessionError;

#[derive(Clone, Deserialize, Debug)]
#[serde(untagged)]
pub enum TokenConfig {
    Value(String),
    Property {
        content: String,
    },
}

#[derive(Clone, Deserialize, Debug)]
pub struct TokenizerConfig {
    pub eos_token: TokenConfig,
    pub chat_template: Option<String>,
}

impl TokenizerConfig {
    pub fn load(model_path: PathBuf) -> Result<Self, SessionError> {
        let tokenizer_config_path = model_path.join("tokenizer_config.json");
        let tokenizer_config_file = File::open(&tokenizer_config_path)
            .map_err(|_| crate::session::session_error::SessionError::UnableToLoadTokenizerConfig)?;
        let mut tokenizer_config: TokenizerConfig =
            serde_json::from_reader(BufReader::new(tokenizer_config_file))
                .map_err(|_| crate::session::session_error::SessionError::UnableToLoadTokenizerConfig)?;

        if tokenizer_config.chat_template.is_none() {
            let chat_template_path = model_path.join("chat_template.jinja");
            if chat_template_path.exists() {
                tokenizer_config.chat_template = Some(
                    std::fs::read_to_string(&chat_template_path).map_err(|_| {
                        crate::session::session_error::SessionError::UnableToLoadTokenizerConfig
                    })?,
                );
            }
        }

        return Ok(tokenizer_config);
    }

    pub fn eos_token(&self) -> String {
        match &self.eos_token {
            TokenConfig::Value(value) => value.clone(),
            TokenConfig::Property {
                content,
            } => content.clone(),
        }
    }
}
