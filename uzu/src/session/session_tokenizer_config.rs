use std::path::PathBuf;

use tokenizers::Tokenizer;

use crate::session::config::generation_metadata::GenerationMetadata;

#[derive(Clone, Debug)]
pub struct SessionTokenizerConfig {
    pub eos_tokens: Vec<String>,
    pub chat_template: String,
}

impl SessionTokenizerConfig {
    pub fn load(
        model_path: PathBuf,
        tokenizer: &Tokenizer,
    ) -> Option<Self> {
        let generation_metadata = GenerationMetadata::load(model_path);

        let eos_tokens =
            Self::build_eos_tokens(tokenizer, &generation_metadata);
        if eos_tokens.is_empty() {
            return None;
        }

        let chat_template = Self::build_chat_template(&generation_metadata);
        if chat_template.len() == 0 {
            return None;
        }

        Some(Self {
            eos_tokens,
            chat_template,
        })
    }

    fn build_eos_tokens(
        tokenizer: &Tokenizer,
        generation_metadata: &GenerationMetadata,
    ) -> Vec<String> {
        let mut eos_tokens: Vec<String> = vec![];

        if let Some(tokenizer_config) = &generation_metadata.tokenizer_config {
            if let Some(eos_token) = &tokenizer_config.eos_token {
                eos_tokens.push(eos_token.to_string());
            }
        }

        eos_tokens
    }

    fn build_chat_template(generation_metadata: &GenerationMetadata) -> String {
        if let Some(tokenizer_config) = &generation_metadata.tokenizer_config {
            if let Some(chat_template) = &tokenizer_config.chat_template {
                return chat_template.clone();
            }
        }

        if let Some(chat_template) = &generation_metadata.chat_template {
            return chat_template.clone();
        }

        "".to_string()
    }
}
