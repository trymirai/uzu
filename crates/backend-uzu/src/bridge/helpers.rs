use shoji::types::{
    basic::{
        ContextLength, Grammar as ShojiGrammar, SamplingMethod as ShojiSamplingMethod,
        SamplingPolicy as ShojiSamplingPolicy,
    },
    session::chat::ChatSpeculationPreset,
};

use crate::{
    backends::common::Backend,
    encodable_block::sampling::{SamplingMethod as UzuSamplingMethod, SamplingProcessingOrder},
    engine::language_model::{LanguageModel, grammar::CompiledGrammar, stream::LanguageModelStreamSpeculatorOptions},
};

pub fn get_grammar<'a, B: Backend>(_grammar: Option<ShojiGrammar>) -> Option<&'a mut dyn CompiledGrammar> {
    // TODO agolokoz: implement
    // grammar.map(|grammar| {
    //     let config = match grammar {
    //         Grammar::JsonAny {
    //             ..
    //         } => GrammarConfig::builtin_json(),
    //         Grammar::JsonSchema {
    //             schema,
    //         } => GrammarConfig::json_schema_simple(schema),
    //         Grammar::Regex {
    //             pattern,
    //         } => GrammarConfig::regex(pattern, false),
    //     };
    // })
    None
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
                processing_order: SamplingProcessingOrder::TemperatureThenFilters,
            },
        },
    }
}

pub fn get_speculator<'a>(
    _speculation_preset: Option<ChatSpeculationPreset>
) -> Option<LanguageModelStreamSpeculatorOptions<'a>> {
    // TODO agolokoz: implement
    // speculation_preset.map(|speculation| {
    // })
    None
}
