use std::pin::Pin;

use futures::{Stream, stream};
#[cfg(feature = "capability-grammar")]
use shoji::types::basic::Grammar as ShojiGrammar;
use shoji::{
    traits::backend::{Error as BackendError, chat_token::StreamOutput as ChatTokenStreamOutput},
    types::basic::{ContextLength, SamplingMethod as ShojiSamplingMethod, SamplingPolicy as ShojiSamplingPolicy},
};
#[cfg(feature = "capability-grammar")]
use tokenizers::Tokenizer;
#[cfg(feature = "capability-grammar")]
use uzu_engine::engine::language_model::grammar::{Grammar as UzuGrammar, GrammarConfig, GrammarError};
use uzu_engine::{
    backends::common::Backend,
    engine::language_model::{LanguageModel, stream::SamplingMethod as UzuSamplingMethod},
};

pub fn error_stream<'a>(
    message: String
) -> Pin<Box<dyn Stream<Item = Result<ChatTokenStreamOutput, BackendError>> + Send + 'a>> {
    Box::pin(stream::once(async move {
        Err::<ChatTokenStreamOutput, BackendError>(Box::<dyn std::error::Error + Send + Sync>::from(message))
    }))
}

#[cfg(feature = "capability-grammar")]
pub fn get_grammar(
    grammar: ShojiGrammar,
    tokenizer: &Tokenizer,
    stop_token_ids: &[i32],
    trigger_token_sequence: Option<Vec<u64>>,
) -> Result<UzuGrammar, GrammarError> {
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
    UzuGrammar::new(&config, tokenizer, trigger_token_sequence, Some(stop_token_ids))
}

/// The token sequence of the model's end-of-thinking tag. A reasoning model
/// opens every reply with reasoning tokens that no grammar can accept, so the
/// grammar must stay disengaged until this sequence has been generated.
#[cfg(feature = "capability-grammar")]
pub fn grammar_trigger_token_sequence<B: Backend>(model: &LanguageModel<B>) -> Option<Vec<u64>> {
    let tag = model.end_of_thinking_tag()?;
    let token_ids: Vec<u64> =
        model.tokenizer().encode(tag, false).ok()?.get_ids().iter().map(|token_id| u64::from(*token_id)).collect();
    (!token_ids.is_empty()).then_some(token_ids)
}

pub fn get_max_context_length<B: Backend>(
    model: &LanguageModel<B>,
    context_length: ContextLength,
) -> Option<u32> {
    match context_length {
        ContextLength::Default {
            ..
        } => model.recommended_context_length(),
        ContextLength::Maximal {
            ..
        } => model.max_context_length(),
        ContextLength::Custom {
            length,
        } => Some(length.max(0) as u32),
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
                suffix_repetition_length: suffix_repetition_length.map(|value| value as u32),
            },
        },
    }
}
