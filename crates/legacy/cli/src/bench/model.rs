use anyhow::{Result, anyhow, bail, ensure};
use rocket::serde::{Deserialize, Serialize};
use uzu::types::{
    basic::{ContextLength, ReasoningEffort, SamplingMethod},
    session::chat::{ChatMessage, ChatReplyEnergy, ChatRole},
};
use uzu_engine::data_type::DataType;

use crate::{
    common::thinking::ThinkingSupport,
    server::{
        chat_completions::{OaiMessage, to_chat_messages},
        chat_tool_calls::{OaiTool, backfill_tool_result_names, choose_tools, insert_tools_message},
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchTask {
    pub identifier: String,
    pub repo_id: String,
    pub number_of_runs: u64,
    pub tokens_limit: u64,
    pub messages: Vec<BenchMessage>,
    pub greedy: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_size: Option<BenchContextSize>,
    #[serde(default = "default_context_padding")]
    pub context_padding: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<BenchGenerationConfig>,
    #[serde(default)]
    pub requires_generation_config: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
}

impl BenchTask {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.number_of_runs > 0, "number_of_runs must be positive");
        ensure!(self.tokens_limit > 0 && self.tokens_limit <= u32::MAX as u64, "tokens_limit must fit a positive u32");
        if let Some(BenchContextSize::Tokens(length)) = self.context_size {
            ensure!(length > 0, "context_size must be positive");
        }
        ensure!(!self.requires_generation_config || self.generation_config.is_some(), "generation_config is required");
        if let Some(config) = &self.generation_config {
            config.validate()?;
        }
        Ok(())
    }

    pub fn context_length(&self) -> Result<ContextLength> {
        Ok(match self.context_size {
            Some(BenchContextSize::Tokens(length)) => ContextLength::Custom {
                length: i64::from(length),
            },
            Some(BenchContextSize::Auto(_)) => bail!("Auto context must be resolved before creating a session"),
            None => ContextLength::Default {},
        })
    }

    pub fn resolve_context_size(
        &self,
        prompt_tokens: u64,
    ) -> Result<Option<u32>> {
        let required = prompt_tokens.checked_add(self.tokens_limit).ok_or_else(|| anyhow!("Context size overflow"))?;
        let size = match self.context_size {
            Some(BenchContextSize::Tokens(length)) => length,
            Some(BenchContextSize::Auto(_)) => u32::try_from(
                required
                    .checked_add(u64::from(self.context_padding))
                    .ok_or_else(|| anyhow!("Context size overflow"))?,
            )?,
            None => return Ok(None),
        };
        ensure!(u64::from(size) >= required, "context_size cannot fit the prompt and output token limit");
        Ok(Some(size))
    }

    pub fn to_chat_messages(
        &self,
        thinking_support: ThinkingSupport,
    ) -> Result<Vec<ChatMessage>> {
        let messages: Vec<OaiMessage> = serde_json::from_value(serde_json::to_value(&self.messages)?)
            .map_err(|_| anyhow!("Invalid benchmark messages"))?;
        let tools: Vec<OaiTool> =
            serde_json::from_value(serde_json::to_value(self.tools.as_deref().unwrap_or_default())?)
                .map_err(|_| anyhow!("Invalid benchmark tools"))?;
        // Reject arguments that the server converter would repair or wrap;
        // replay must preserve the recorded calls.
        for call in messages.iter().flat_map(|message| message.tool_calls.iter().flatten()) {
            ensure!(
                serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                    .is_ok_and(|arguments| arguments.is_object()),
                "Invalid benchmark tool arguments: expected a JSON object"
            );
        }
        let mut messages = to_chat_messages(&messages);
        if let Some(reasoning) = self.reasoning {
            let requested = if reasoning {
                ReasoningEffort::Default
            } else {
                ReasoningEffort::Disabled
            };
            if let Some(effort) = thinking_support.fulfill_requested_effort(requested).map_err(anyhow::Error::msg)? {
                match messages.first_mut() {
                    Some(first) if first.role == (ChatRole::System {}) => {
                        *first = first.clone().with_reasoning_effort(effort)
                    },
                    _ => messages.insert(0, ChatMessage::system().with_reasoning_effort(effort)),
                }
            }
        }
        backfill_tool_result_names(&mut messages);
        let tools = choose_tools(Some(&tools), self.tool_choice.as_ref())
            .map_err(|_| anyhow!("Invalid benchmark tool_choice"))?;
        insert_tools_message(&mut messages, &tools);
        Ok(messages)
    }
}

fn default_context_padding() -> u32 {
    64
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BenchContextSize {
    Tokens(u32),
    Auto(AutoContextSize),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AutoContextSize {
    Auto,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchGenerationConfig {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_token_ids: Option<Vec<u64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suffix_repetition_length: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub banned_tokens: Option<Vec<u64>>,
}

impl BenchGenerationConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (self.temperature as f32).is_finite() && (self.temperature as f32) > 0.0,
            "temperature must be finite and positive"
        );
        ensure!((self.top_p as f32) > 0.0 && self.top_p <= 1.0, "top_p must be in (0, 1]");
        ensure!(self.min_p.is_none_or(|value| (0.0..=1.0).contains(&value)), "min_p must be in [0, 1]");
        ensure!(
            self.repetition_penalty.is_none_or(|value| value == 1.0),
            "Nonneutral repetition_penalty is unsupported by speculative benchmarks"
        );
        ensure!(
            self.suffix_repetition_length.is_none_or(|value| value == 0),
            "suffix_repetition_length is unsupported by speculative benchmarks"
        );
        ensure!(self.presence_penalty.is_none_or(|value| value == 0.0), "presence_penalty is unsupported");
        ensure!(self.frequency_penalty.is_none_or(|value| value == 0.0), "frequency_penalty is unsupported");
        ensure!(self.banned_tokens.as_ref().is_none_or(Vec::is_empty), "banned_tokens is unsupported");
        Ok(())
    }

    pub fn validate_stop_tokens(
        &self,
        actual: Option<&[u64]>,
    ) -> Result<()> {
        ensure!(
            self.stop_token_ids.as_deref().is_none_or(|requested| Some(requested) == actual),
            "Requested stop_token_ids do not match the model; per-request stop overrides are unsupported"
        );
        Ok(())
    }

    pub fn sampling_method(&self) -> SamplingMethod {
        SamplingMethod::Stochastic {
            temperature: Some(self.temperature),
            top_k: (self.top_k > 0).then_some(i64::from(self.top_k)),
            top_p: Some(self.top_p),
            min_p: self.min_p,
            repetition_penalty: None,
            suffix_repetition_length: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BenchMessage {
    pub role: BenchMessageRole,
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum BenchMessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchDevice {
    pub os_name: Option<String>,
    pub cpu_name: Option<String>,
    pub memory_total: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    pub task: BenchTask,
    pub device: BenchDevice,
    pub engine_version: String,
    pub timestamp: u64,
    pub data_type: DataType,
    pub memory_used: Option<usize>,
    pub tokens_count_input: u64,
    pub tokens_count_output: u64,
    pub time_to_first_token: f64,
    pub prompt_tokens_per_second: f64,
    pub generate_tokens_per_second: Option<f64>,
    pub input_energy: Option<ChatReplyEnergy>,
    pub output_energy: Option<ChatReplyEnergy>,
    pub joules_per_token: Option<f64>,
    pub text: String,
}

#[cfg(test)]
#[path = "../../unit/bench/model_test.rs"]
mod tests;
