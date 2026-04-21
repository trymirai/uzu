use std::{path::Path, sync::Arc};

use shoji::{
    traits::backend::chat_message::Output as ChatMessageOutput,
    types::{
        basic::{
            SamplingMethod as ShojiSamplingMethod, SamplingPolicy as ShojiSamplingPolicy,
            SamplingSeed as ShojiSamplingSeed,
        },
        encoding::{Message as ShojiMessage, MessageList, ReasoningEffort, Role as ShojiRole},
        session::chat::{
            Config as ShojiChatConfig, ContextLength as ShojiContextLength, FinishReason as ShojiFinishReason,
            Grammar as ShojiGrammar, SpeculationPreset as ShojiSpeculationPreset, Stats as ShojiStats, StreamConfig,
        },
    },
};
use tokenizers::Tokenizer;

use crate::{
    inference::Error,
    session::{
        config::{DecodingConfig, GrammarConfig, RunConfig, SpeculatorConfig},
        parameter::{ContextLength, SamplingMethod, SamplingPolicy, SamplingProcessingOrder, SamplingSeed},
        types::{FinishReason, Input, Message, Output, Role, Stats},
    },
    speculators::{fixed_token_speculator::FixedTokensSpeculator, prompt_lookup_speculator::PromptLookupSpeculator},
};

pub fn build_decoding_config(
    config: &ShojiChatConfig,
    model_path: &Path,
) -> Result<DecodingConfig, Error> {
    let context_length = match &config.context_length {
        ShojiContextLength::Default {} => ContextLength::Default,
        ShojiContextLength::Maximal {} => ContextLength::Maximal,
        ShojiContextLength::Custom {
            length,
        } => ContextLength::Custom((*length).max(0) as usize),
    };
    let sampling_seed = match &config.sampling_seed {
        ShojiSamplingSeed::Default {} => SamplingSeed::Default,
        ShojiSamplingSeed::Custom {
            seed,
        } => SamplingSeed::Custom(*seed as u64),
    };
    let speculator_config: SpeculatorConfig = match &config.speculation_preset {
        Some(ShojiSpeculationPreset::GeneralChat {
            ..
        }) => SpeculatorConfig::default(),
        Some(ShojiSpeculationPreset::Summarization {
            ..
        }) => {
            let number_of_speculated_tokens = 16 - 1;
            let speculator = PromptLookupSpeculator::new_with_params(3);
            SpeculatorConfig::new(number_of_speculated_tokens, Arc::new(speculator))
        },
        Some(ShojiSpeculationPreset::Classification {
            feature,
        }) => {
            let tokenizer =
                Tokenizer::from_file(model_path.join("tokenizer.json")).map_err(|_| Error::UnableToLoadTokenizer)?;
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
            let speculator = FixedTokensSpeculator::new(proposals);
            SpeculatorConfig::new(speculator.max_trie_nodes(), Arc::new(speculator))
        },
        None => SpeculatorConfig::default(),
    };
    Ok(DecodingConfig::default()
        .with_context_length(context_length)
        .with_sampling_seed(sampling_seed)
        .with_speculator_config(speculator_config))
}

pub fn build_input_and_run_config(
    messages: &Vec<ShojiMessage>,
    config: &StreamConfig,
) -> (Input, RunConfig) {
    let input_messages = messages.iter().filter_map(build_message).collect();
    let input = Input::Messages(input_messages);
    let run_config = build_run_config(messages, config);
    (input, run_config)
}

pub fn build_output(output: &Output) -> ChatMessageOutput {
    let text = output.text.parsed.response.clone();
    let reasoning = output.text.parsed.chain_of_thought.clone();
    let finish_reason = output.finish_reason.as_ref().map(build_finish_reason);
    let stats = build_stats(&output.stats);
    ChatMessageOutput {
        reasoning,
        text,
        tool_calls: vec![],
        finish_reason,
        stats,
    }
}

fn build_message(message: &ShojiMessage) -> Option<Message> {
    let role = match &message.role {
        ShojiRole::System {} => Role::System,
        ShojiRole::User {} => Role::User,
        ShojiRole::Assistant {} => Role::Assistant,
        ShojiRole::Developer {}
        | ShojiRole::Tool {}
        | ShojiRole::Custom {
            ..
        } => return None,
    };

    let content = message.text().unwrap_or_default();
    let reasoning_content = message.reasoning();
    Some(Message {
        role,
        content,
        reasoning_content,
    })
}

fn build_run_config(
    messages: &Vec<ShojiMessage>,
    config: &StreamConfig,
) -> RunConfig {
    let sampling_policy = build_sampling_policy(&config.sampling_policy);
    let grammar_config = build_grammar(&config.grammar);
    let tokens_limit = config.token_limit.map(|value| value as u64).unwrap_or(2048);
    let enable_thinking =
        messages.reasoning_effort().map(|effort| !matches!(effort, ReasoningEffort::Disabled)).unwrap_or(true);
    RunConfig::new(tokens_limit, enable_thinking, sampling_policy, grammar_config)
}

fn build_sampling_policy(sampling_policy: &ShojiSamplingPolicy) -> SamplingPolicy {
    match sampling_policy {
        ShojiSamplingPolicy::Default {} => SamplingPolicy::Default,
        ShojiSamplingPolicy::Custom {
            method,
        } => SamplingPolicy::Custom {
            value: match method {
                ShojiSamplingMethod::Greedy {} => SamplingMethod::Greedy,
                ShojiSamplingMethod::Stochastic {
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                } => SamplingMethod::Stochastic {
                    temperature: temperature.map(|value| value as f32),
                    top_k: top_k.map(|value| value.max(0) as u32),
                    top_p: top_p.map(|value| value as f32),
                    min_p: min_p.map(|value| value as f32),
                    processing_order: SamplingProcessingOrder::TemperatureThenFilters,
                },
            },
        },
    }
}

fn build_grammar(grammar: &Option<ShojiGrammar>) -> Option<GrammarConfig> {
    match grammar {
        Some(ShojiGrammar::JsonAny {}) => Some(GrammarConfig::builtin_json()),
        Some(ShojiGrammar::JsonSchema {
            schema,
        }) => Some(GrammarConfig::json_schema_simple(schema.clone())),
        Some(ShojiGrammar::Regex {
            pattern,
        }) => Some(GrammarConfig::regex(pattern.clone(), false)),
        None => None,
    }
}

fn build_finish_reason(reason: &FinishReason) -> ShojiFinishReason {
    match reason {
        FinishReason::Stop => ShojiFinishReason::Stop,
        FinishReason::Length => ShojiFinishReason::Length,
        FinishReason::Cancelled => ShojiFinishReason::Cancelled,
        FinishReason::ContextLimitReached => ShojiFinishReason::ContextLimitReached,
    }
}

fn build_stats(stats: &Stats) -> ShojiStats {
    let time_to_first_token = Some(stats.prefill_stats.duration);
    let prefill_tokens_per_second = Some(stats.prefill_stats.processed_tokens_per_second);
    let generate_tokens_per_second = stats.generate_stats.as_ref().map(|step| step.tokens_per_second);
    ShojiStats {
        duration: stats.total_stats.duration,
        time_to_first_token,
        prefill_tokens_per_second,
        generate_tokens_per_second,
        tokens_count_input: Some(stats.total_stats.tokens_count_input as u32),
        tokens_count_output: Some(stats.total_stats.tokens_count_output as u32),
    }
}
