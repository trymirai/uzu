use std::{collections::HashMap, path::PathBuf};

use tokenizers::{AddedToken, Tokenizer};

use crate::session::{
    config::generation_metadata::GenerationMetadata,
    sampling_config::SamplingConfig,
};

#[derive(Clone, Debug)]
pub struct SessionTokenizerConfig {
    pub eos_tokens: Vec<String>,
    pub chat_template: String,
    pub sampling_config: SamplingConfig,
}

impl SessionTokenizerConfig {
    pub fn load_and_add_special_tokens_to_tokenizer(
        model_path: PathBuf,
        tokenizer: &mut Tokenizer,
    ) -> Option<Self> {
        let generation_metadata = GenerationMetadata::load(model_path);
        Self::add_special_tokens_to_tokenizer_from_metadata(
            tokenizer,
            &generation_metadata,
        );

        let eos_tokens =
            Self::build_eos_tokens(tokenizer, &generation_metadata);
        if eos_tokens.is_empty() {
            return None;
        }

        let chat_template = Self::build_chat_template(&generation_metadata);
        if chat_template.len() == 0 {
            return None;
        }

        let sampling_config = Self::build_sampling_config(&generation_metadata);

        Some(Self {
            eos_tokens,
            chat_template,
            sampling_config,
        })
    }

    fn add_special_tokens_to_tokenizer_from_metadata(
        tokenizer: &mut Tokenizer,
        generation_metadata: &GenerationMetadata,
    ) {
        if let Some(tokenizer_config) = &generation_metadata.tokenizer_config {
            if let Some(added_tokens_decoder) =
                &tokenizer_config.added_tokens_decoder
            {
                if let Some(additional_special_tokens) =
                    &tokenizer_config.additional_special_tokens
                {
                    Self::add_special_tokens_to_tokenizer_from_list(
                        tokenizer,
                        added_tokens_decoder,
                        additional_special_tokens.clone(),
                    );
                }

                if let Some(extra_special_tokens) =
                    &tokenizer_config.extra_special_tokens
                {
                    Self::add_special_tokens_to_tokenizer_from_list(
                        tokenizer,
                        added_tokens_decoder,
                        extra_special_tokens
                            .values()
                            .cloned()
                            .collect::<Vec<_>>(),
                    );
                }
            }
        }
    }

    fn add_special_tokens_to_tokenizer_from_list(
        tokenizer: &mut Tokenizer,
        added_tokens_decoder: &HashMap<u32, AddedToken>,
        special_tokens: Vec<String>,
    ) {
        let special_added_tokens = added_tokens_decoder
            .iter()
            .filter(|(_, token)| special_tokens.contains(&token.content))
            .map(|(_, token)| token.clone())
            .collect::<Vec<_>>();
        tokenizer.add_special_tokens(&special_added_tokens);
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

        if let Some(generation_config) = &generation_metadata.generation_config
        {
            if let Some(eos_token_id) = &generation_config.eos_token_id {
                let eos_token_id_list = eos_token_id.to_list();
                let eos_token_list = eos_token_id_list
                    .iter()
                    .flat_map(|token_id| tokenizer.id_to_token(*token_id))
                    .collect::<Vec<_>>();
                eos_tokens.extend(eos_token_list);
            }
        }

        eos_tokens.sort();
        eos_tokens.dedup();
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

    fn build_sampling_config(
        generation_metadata: &GenerationMetadata
    ) -> SamplingConfig {
        if let Some(generation_config) = &generation_metadata.generation_config
        {
            if let Some(top_p) = &generation_config.top_p {
                return SamplingConfig::TopP {
                    top_p: *top_p,
                };
            }
            if let Some(temperature) = &generation_config.temperature {
                return SamplingConfig::Categorical {
                    temperature: *temperature,
                };
            }
        }
        return SamplingConfig::Argmax;
    }
}
