mod bundled;
mod tokens;

use serde::{Deserialize, Serialize};
use shoji::types::encoding::{Capabilities, ContentBlockType, ReasoningEffort};
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "name", rename_all = "snake_case")]
pub enum Config {
    #[serde(rename = "gpt-oss")]
    GptOss,
    #[serde(rename = "llama-3.1")]
    Llama31,
    #[serde(rename = "llama-3.2")]
    Llama32,
    #[serde(rename = "qwen2.5")]
    Qwen25,
    #[serde(rename = "qwen2.5-coder")]
    Qwen25Coder,
    #[serde(rename = "qwen3")]
    Qwen3,
    #[serde(rename = "qwen3-instruct")]
    Qwen3Instruct,
    #[serde(rename = "qwen3-thinking")]
    Qwen3Thinking,
    #[serde(rename = "qwen3.5")]
    Qwen35,
    #[serde(rename = "lfm2")]
    Lfm2,
    #[serde(rename = "lfm2.5-instruct")]
    Lfm25Instruct,
    #[serde(rename = "lfm2.5-thinking")]
    Lfm25Thinking,
    #[serde(rename = "deepseek-r1-distill-qwen")]
    DeepseekR1DistillQwen,
    #[serde(rename = "llamba")]
    Llamba,
    #[serde(rename = "rnj-1")]
    Rnj1,
    #[serde(rename = "gemma-2")]
    Gemma2,
    #[serde(rename = "gemma-3")]
    Gemma3,
    #[serde(rename = "gemma-4")]
    Gemma4,
    #[serde(rename = "functiongemma")]
    FunctionGemma,
    #[serde(rename = "translategemma")]
    TranslateGemma,
    #[serde(rename = "smollm2")]
    SmolLm2,
    #[serde(rename = "smollm3")]
    SmolLm3,
    #[serde(rename = "codestral")]
    Codestral,
    #[serde(rename = "mistral-small-3")]
    MistralSmall3,
    #[serde(rename = "devstral-small")]
    DevstralSmall,
    #[serde(rename = "polaris")]
    Polaris,
    #[serde(rename = "bonsai")]
    Bonsai,
    #[serde(rename = "nanbeige")]
    Nanbeige,
    #[serde(rename = "custom")]
    Custom {
        #[serde(flatten)]
        config: ResolvedConfig,
    },
}

impl Config {
    pub fn capabilities(&self) -> Result<Capabilities, Error> {
        let resolved = self.clone().resolve()?;
        let rendering = &resolved.rendering;

        let supports_multiple_tool_calls = rendering
            .get_rendering_field_for_block_type(&ContentBlockType::ToolCall)
            .and_then(|field| match &field.config {
                FieldConfig::Collected {
                    limit,
                    ..
                } => *limit,
                _ => None,
            })
            .map_or(true, |limit| limit > 1);

        let supports_disable_reasoning = rendering
            .get_rendering_field_for_block_type(&ContentBlockType::ReasoningEffort)
            .map_or(false, |field| match &field.config {
                FieldConfig::Unique {
                    mapping: Some(mapping),
                    ..
                } => mapping.contains_key(ReasoningEffort::Disabled.to_string().as_str()),
                _ => false,
            });

        let tools_role_and_field = rendering.get_rendering_role_and_field_for_block_type(&ContentBlockType::Tools);
        let supports_tools = tools_role_and_field.is_some();
        let requires_tools = tools_role_and_field
            .map(|(role, field)| field.required && !resolved.ordering.is_role_avoidable(role))
            .unwrap_or(false);

        Ok(Capabilities {
            supports_reasoning: rendering.get_rendering_field_for_block_type(&ContentBlockType::Reasoning).is_some(),
            supports_disable_reasoning,
            supports_tools,
            supports_multiple_tool_calls,
            requires_tools,
        })
    }

    pub fn resolve(self) -> Result<ResolvedConfig, Error> {
        match self {
            Config::GptOss => resolve_bundled_config("gpt-oss"),
            Config::Llama31 => resolve_bundled_config("llama-3.1"),
            Config::Llama32 => resolve_bundled_config("llama-3.2"),
            Config::Qwen25 => resolve_bundled_config("qwen2.5"),
            Config::Qwen25Coder => resolve_bundled_config("qwen2.5-coder"),
            Config::Qwen3 => resolve_bundled_config("qwen3"),
            Config::Qwen3Instruct => resolve_bundled_config("qwen3-instruct"),
            Config::Qwen3Thinking => resolve_bundled_config("qwen3-thinking"),
            Config::Qwen35 => resolve_bundled_config("qwen3.5"),
            Config::Lfm2 => resolve_bundled_config("lfm2"),
            Config::Lfm25Instruct => resolve_bundled_config("lfm2.5-instruct"),
            Config::Lfm25Thinking => resolve_bundled_config("lfm2.5-thinking"),
            Config::DeepseekR1DistillQwen => resolve_bundled_config("deepseek-r1-distill-qwen"),
            Config::Llamba => resolve_bundled_config("llama-3.2"),
            Config::Rnj1 => resolve_bundled_config("rnj-1"),
            Config::Gemma2 => resolve_bundled_config("gemma-2"),
            Config::Gemma3 => resolve_bundled_config("gemma-3"),
            Config::Gemma4 => resolve_bundled_config("gemma-4"),
            Config::FunctionGemma => resolve_bundled_config("functiongemma"),
            Config::TranslateGemma => resolve_bundled_config("translategemma"),
            Config::SmolLm2 => resolve_bundled_config("smollm2"),
            Config::SmolLm3 => resolve_bundled_config("smollm3"),
            Config::Codestral => resolve_bundled_config("codestral"),
            Config::MistralSmall3 => resolve_bundled_config("mistral-small-3"),
            Config::DevstralSmall => resolve_bundled_config("devstral-small"),
            Config::Polaris => resolve_bundled_config("polaris"),
            Config::Bonsai => resolve_bundled_config("qwen3"),
            Config::Nanbeige => resolve_bundled_config("nanbeige"),
            Config::Custom {
                config,
            } => Ok(config),
        }
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
