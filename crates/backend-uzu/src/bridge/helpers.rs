use std::pin::Pin;

use futures::{Stream, stream};
use shoji::{
    traits::backend::{Error as BackendError, chat_token::StreamOutput as ChatTokenStreamOutput},
    types::{
        basic::{
            ContextLength, Grammar as ShojiGrammar, SamplingMethod as ShojiSamplingMethod,
            SamplingPolicy as ShojiSamplingPolicy,
        },
        session::chat::ChatSpeculationPreset,
    },
};
use tokenizers::Tokenizer;

use crate::{
    backends::common::Backend,
    encodable_block::sampling::SamplingMethod as UzuSamplingMethod,
    engine::language_model::{
        LanguageModel,
        grammar::{Grammar as UzuGrammar, GrammarConfig, GrammarError},
    },
    speculators::{
        fixed_token_speculator::FixedTokensSpeculator, prompt_lookup_speculator::PromptLookupSpeculator,
        speculator::Speculator,
    },
};

pub fn error_stream<'a>(
    message: String
) -> Pin<Box<dyn Stream<Item = Result<ChatTokenStreamOutput, BackendError>> + Send + 'a>> {
    Box::pin(stream::once(async move {
        Err::<ChatTokenStreamOutput, BackendError>(Box::<dyn std::error::Error + Send + Sync>::from(message))
    }))
}

pub fn get_grammar<'a, B: Backend>(
    grammar: ShojiGrammar,
    tokenizer: &Tokenizer,
    stop_token_ids: &[i32],
) -> Result<Box<dyn UzuGrammar>, GrammarError> {
    let config = match grammar {
        ShojiGrammar::JsonAny {
            ..
        } => GrammarConfig::builtin_json(),
        ShojiGrammar::JsonSchema {
            schema,
        } => GrammarConfig::json_schema_simple(schema),
        ShojiGrammar::Regex {
            pattern,
        } => GrammarConfig::regex(pattern, false),
    };
    <dyn UzuGrammar>::new(&config, tokenizer, None, Some(stop_token_ids))
}

pub fn get_max_context_length<B: Backend>(
    model: &LanguageModel<B>,
    context_length: ContextLength,
) -> Option<usize> {
    match context_length {
        ContextLength::Default {
            ..
        } => model.recommended_context_length(),
        ContextLength::Maximal {
            ..
        } => model.max_context_length(),
        ContextLength::Custom {
            length,
        } => Some(length as usize),
    }
}

pub fn get_sampling_method<B: Backend>(
    model: &LanguageModel<B>,
    sampling_method: &ShojiSamplingPolicy,
) -> UzuSamplingMethod {
    match sampling_method {
        ShojiSamplingPolicy::Default {
            ..
        } => model.default_sampling_method(),
        ShojiSamplingPolicy::Custom {
            method,
        } => match method {
            ShojiSamplingMethod::Greedy {
                ..
            } => UzuSamplingMethod::Greedy,
            ShojiSamplingMethod::Stochastic {
                temperature,
                top_k,
                top_p,
                min_p,
                repetition_penalty,
                suffix_repetition_length,
            } => UzuSamplingMethod::Stochastic {
                temperature: temperature.map(|value| value as f32),
                top_k: top_k.map(|value| value as u32),
                top_p: top_p.map(|value| value as f32),
                min_p: min_p.map(|value| value as f32),
                repetition_penalty: repetition_penalty.map(|value| value as f32),
                suffix_repetition_length: suffix_repetition_length.map(|value| value as usize),
            },
        },
    }
}

pub fn get_speculator<'a>(
    preset: &ChatSpeculationPreset,
    tokenizer: &Tokenizer,
) -> Result<Option<(Box<dyn Speculator>, usize)>, BackendError> {
    match preset {
        ChatSpeculationPreset::GeneralChat {
            ..
        } => Ok(None),
        ChatSpeculationPreset::Summarization {
            ..
        } => {
            let speculator = Box::new(PromptLookupSpeculator::new_with_params(3));
            Ok(Some((speculator, 16)))
        },
        ChatSpeculationPreset::Classification {
            feature,
        } => {
            let proposals = feature
                .values
                .iter()
                .map(|value| {
                    let encoding = tokenizer.encode(value.as_str(), false).map_err(|err| BackendError::from(err))?;
                    let ids = encoding.get_ids().iter().map(|&id| id as u64).collect::<Vec<_>>();
                    Ok(ids)
                })
                .collect::<Result<Vec<Vec<u64>>, BackendError>>()?;
            let speculator = Box::new(FixedTokensSpeculator::new(proposals));
            let budget = speculator.max_trie_nodes() + 1;
            Ok(Some((speculator, budget)))
        },
    }
}
