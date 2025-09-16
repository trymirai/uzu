use tokenizers::Tokenizer;

use crate::{
    backends::metal::sampling_config::SamplingConfig,
    config::{GenerationConfig, LanguageModelConfig},
};

#[derive(Clone, Debug)]
pub struct SessionTokenizerConfig {
    pub eos_tokens: Vec<String>,
    pub bos_token: Option<String>,
    pub chat_template: String,
    pub sampling_config: SamplingConfig,
}

impl SessionTokenizerConfig {
    pub fn load(
        model_config: &LanguageModelConfig,
        tokenizer: &Tokenizer,
    ) -> Option<Self> {
        let eos_tokens = model_config
            .generation_config
            .stop_token_ids
            .iter()
            .flat_map(|token_id| tokenizer.id_to_token(*token_id))
            .collect::<Vec<_>>();
        if eos_tokens.is_empty() {
            return None;
        }

        let bos_token = model_config.message_processor_config.bos_token.clone();

        let chat_template =
            model_config.message_processor_config.prompt_template.clone();

        let sampling_config =
            Self::build_sampling_config(&model_config.generation_config);

        Some(Self {
            eos_tokens,
            bos_token,
            chat_template,
            sampling_config,
        })
    }

    fn build_sampling_config(
        generation_config: &GenerationConfig
    ) -> SamplingConfig {
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
        return SamplingConfig::Argmax;
    }
}
