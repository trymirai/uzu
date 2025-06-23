use tokenizers::Tokenizer;

use super::{
    sampling_config::SamplingConfig,
    session_classification_feature::SessionClassificationFeature,
};
use crate::{
    generator::config::{
        ContextLength, GeneratorConfig, SamplingSeed, SpeculatorConfig,
    },
    linearizer::trie::TokenTrie,
    speculators::{
        empty_speculator::EmptySpeculator,
        fixed_token_speculator::FixedTokensSpeculator,
        prompt_lookup_speculator::PromptLookupSpeculator,
    },
};

#[derive(Debug)]
pub enum SessionPreset {
    General,
    Classification(SessionClassificationFeature),
    Summarization,
}

#[derive(Debug)]
pub struct SessionConfig {
    pub preset: SessionPreset,
    pub sampling_seed: SamplingSeed,
    pub context_length: ContextLength,
}

impl SessionConfig {
    pub fn new(
        preset: SessionPreset,
        sampling_seed: SamplingSeed,
        context_length: ContextLength,
    ) -> Self {
        Self {
            preset,
            sampling_seed,
            context_length,
        }
    }

    pub fn generator_config(
        &self,
        tokenizer: &Tokenizer,
    ) -> GeneratorConfig {
        let prefill_step_size: usize;
        let prefix_length_step: Option<usize> = None;
        let speculator_config: SpeculatorConfig;
        let allow_pre_encode = true;
        match &self.preset {
            SessionPreset::General => {
                prefill_step_size = 8;
                speculator_config = SpeculatorConfig {
                    number_of_speculated_tokens: 0,
                    speculator: Box::new(EmptySpeculator {}),
                }
            },
            SessionPreset::Classification(feature) => {
                let proposals: Vec<Vec<u64>> = feature
                    .values
                    .iter()
                    .map(|value| {
                        tokenizer
                            .encode(value.clone().as_str(), false)
                            .unwrap()
                            .get_ids()
                            .iter()
                            .map(|&id| id as u64)
                            .collect()
                    })
                    .collect();
                let speculated_suffix = TokenTrie::from_sequences(&proposals)
                    .linearize(0, usize::MAX);

                prefill_step_size = 96;
                speculator_config = SpeculatorConfig {
                    number_of_speculated_tokens: speculated_suffix.tokens.len(),
                    speculator: Box::new(FixedTokensSpeculator::new(proposals)),
                }
            },
            SessionPreset::Summarization => {
                let number_of_speculated_tokens = 16 - 1;
                let speculator = PromptLookupSpeculator::new_with_params(
                    3,
                    number_of_speculated_tokens,
                );

                prefill_step_size = 96;
                speculator_config = SpeculatorConfig {
                    number_of_speculated_tokens: number_of_speculated_tokens,
                    speculator: Box::new(speculator),
                }
            },
        }
        let generator_config = GeneratorConfig::new(
            prefill_step_size,
            prefix_length_step,
            speculator_config,
            allow_pre_encode,
            self.sampling_seed,
            self.context_length,
        );
        generator_config
    }
}

#[derive(Debug)]
pub struct SessionRunConfig {
    pub tokens_limit: u64,
    pub sampling_method: SamplingConfig,
}

impl SessionRunConfig {
    pub fn new(tokens_limit: u64) -> Self {
        Self {
            tokens_limit: tokens_limit,
            sampling_method: SamplingConfig::default(),
        }
    }

    pub fn new_with_sampling(
        tokens_limit: u64,
        sampling_method: SamplingConfig,
    ) -> Self {
        Self {
            tokens_limit: tokens_limit,
            sampling_method,
        }
    }
}
