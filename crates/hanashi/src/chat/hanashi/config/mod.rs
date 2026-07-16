mod bundled;
mod tokens;

use serde::{Deserialize, Serialize};
use shoji::types::{
    basic::ReasoningEffort,
    model::HanashiConfig,
    session::chat::{ChatContentBlockType, ChatModelCapabilities},
};
use token_stream_parser::token_stream::TokenStreamParserConfig;
pub use tokens::TokensConfig;

use crate::chat::hanashi::{
    error::Error, messages::rendered::FieldConfig, ordering::Config as OrderingConfig, renderer::RendererConfig,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ResolvedConfig {
    pub parsing: TokenStreamParserConfig,
    pub rendering: RendererConfig,
    pub tokens: TokensConfig,
    pub ordering: OrderingConfig,
}

pub fn hanashi_config_capabilities(config: &HanashiConfig) -> Result<ChatModelCapabilities, Error> {
    let resolved = hanashi_config_resolve(config)?;
    let rendering = &resolved.rendering;

    let supports_multiple_tool_calls = rendering
        .get_rendering_field_for_block_type(&ChatContentBlockType::ToolCall)
        .and_then(|field| match &field.config {
            FieldConfig::Collected {
                limit,
                ..
            } => *limit,
            _ => None,
        })
        .is_none_or(|limit| limit > 1);

    let supports_disable_reasoning = rendering
        .get_rendering_field_for_block_type(&ChatContentBlockType::ReasoningEffort)
        .is_some_and(|field| match &field.config {
            FieldConfig::Unique {
                mapping: Some(mapping),
                ..
            } => mapping.contains_key(ReasoningEffort::Disabled.to_string().as_str()),
            _ => false,
        });

    let tools_role_and_field = rendering.get_rendering_role_and_field_for_block_type(&ChatContentBlockType::Tools);
    let supports_tools = tools_role_and_field.is_some();
    let requires_tools = tools_role_and_field
        .map(|(role, field)| field.required && !resolved.ordering.is_role_avoidable(role))
        .unwrap_or(false);

    Ok(ChatModelCapabilities {
        supports_reasoning: rendering.get_rendering_field_for_block_type(&ChatContentBlockType::Reasoning).is_some(),
        supports_disable_reasoning,
        supports_tools,
        supports_multiple_tool_calls,
        requires_tools,
    })
}

pub fn hanashi_config_resolve(config: &HanashiConfig) -> Result<ResolvedConfig, Error> {
    match config {
        HanashiConfig::FunctionGemma => resolve_bundled_config("functiongemma"),
        HanashiConfig::Gemma3 => resolve_bundled_config("gemma-3"),
        HanashiConfig::Gemma4 => resolve_bundled_config("gemma-4"),
        HanashiConfig::GptOss => resolve_bundled_config("gpt-oss"),
        HanashiConfig::Lfm2 => resolve_bundled_config("lfm2"),
        HanashiConfig::Lfm25Instruct => resolve_bundled_config("lfm2.5-instruct"),
        HanashiConfig::Lfm25Thinking => resolve_bundled_config("lfm2.5-thinking"),
        HanashiConfig::Llama32 => resolve_bundled_config("llama-3.2"),
        HanashiConfig::Qwen3 => resolve_bundled_config("qwen3"),
        HanashiConfig::Qwen3Instruct => resolve_bundled_config("qwen3-instruct"),
        HanashiConfig::Qwen3Thinking => resolve_bundled_config("qwen3-thinking"),
        HanashiConfig::Qwen35 => resolve_bundled_config("qwen3.5"),
        HanashiConfig::Qwen36 => resolve_bundled_config("qwen3.6"),
    }
}

fn resolve_bundled_config(name: &str) -> Result<ResolvedConfig, Error> {
    let (parsing_name, rendering_name, tokens_name, ordering_name) =
        bundled::get_config_mapping(name).ok_or_else(|| Error::ConfigNotFound(name.to_string()))?;

    let parsing_json =
        bundled::get_parsing_config(parsing_name).ok_or_else(|| Error::ConfigNotFound(parsing_name.to_string()))?;
    let parsing_config =
        serde_json::from_str(parsing_json).map_err(|_| Error::InvalidConfig(parsing_name.to_string()))?;

    let rendering_json = bundled::get_rendering_config(rendering_name)
        .ok_or_else(|| Error::ConfigNotFound(rendering_name.to_string()))?;
    let rendering_config =
        serde_json::from_str(rendering_json).map_err(|_| Error::InvalidConfig(rendering_name.to_string()))?;

    let tokens_json =
        bundled::get_tokens_config(tokens_name).ok_or_else(|| Error::ConfigNotFound(tokens_name.to_string()))?;
    let tokens_config = serde_json::from_str(tokens_json).map_err(|_| Error::InvalidConfig(tokens_name.to_string()))?;

    let ordering_json =
        bundled::get_ordering_config(ordering_name).ok_or_else(|| Error::ConfigNotFound(ordering_name.to_string()))?;
    let ordering_config =
        serde_json::from_str(ordering_json).map_err(|_| Error::InvalidConfig(ordering_name.to_string()))?;

    let config = ResolvedConfig {
        parsing: parsing_config,
        rendering: rendering_config,
        tokens: tokens_config,
        ordering: ordering_config,
    };
    Ok(config)
}
