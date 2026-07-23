use std::pin::Pin;

use futures::{Stream, stream};
use shoji::{
    traits::backend::{Error as BackendError, chat_token::StreamOutput as ChatTokenStreamOutput},
    types::basic::{
        ContextLength, Grammar as ShojiGrammar, SamplingMethod as ShojiSamplingMethod,
        SamplingPolicy as ShojiSamplingPolicy,
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
};

pub fn error_stream<'a>(
    message: String
) -> Pin<Box<dyn Stream<Item = Result<ChatTokenStreamOutput, BackendError>> + Send + 'a>> {
    Box::pin(stream::once(async move {
        Err::<ChatTokenStreamOutput, BackendError>(Box::<dyn std::error::Error + Send + Sync>::from(message))
    }))
}

pub fn get_grammar(
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
        } => Some(length.max(0) as usize),
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
