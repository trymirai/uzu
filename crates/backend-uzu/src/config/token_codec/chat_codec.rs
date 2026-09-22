use std::collections::BTreeMap;

use proc_macros::uzu_config;
use serde::{Deserialize, Serialize};

use crate::utils::strict_serde::DeserializeStrict;

/// Which template fields each reasoning effort sets. Lalamo's renderer consumes this; uzu renders through
/// hanashi and carries it only so one package can be served by both runtimes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReasoningConfig {
    pub default_reasoning_effort: String,
    pub field_name: String,
    pub reasoning_effort_to_field_value: BTreeMap<String, serde_json::Value>,
}

impl<'de> DeserializeStrict<'de> for ReasoningConfig {}

#[uzu_config(super::TokenCodecConfig)]
pub struct ChatCodecConfig {
    pub prompt_template: String,
    pub output_parser_regex: Option<String>,
    pub system_role_name: String,
    pub user_role_name: String,
    pub assistant_role_name: String,
    pub eos_token: Option<String>,
    pub bos_token: Option<String>,
    pub end_of_thinking_tag: Option<String>,
    pub default_system_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_config: Option<ReasoningConfig>,
}
