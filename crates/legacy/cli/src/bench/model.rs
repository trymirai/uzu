use anyhow::{Result, anyhow, ensure};
use rocket::serde::{Deserialize, Serialize};
use uzu::types::session::chat::{ChatMessage, ChatReplyEnergy};
use uzu_engine::data_type::DataType;

use crate::server::{
    chat_completions::{OaiMessage, to_chat_messages},
    chat_tool_calls::{OaiTool, backfill_tool_result_names, choose_tools, insert_tools_message},
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
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
}

impl BenchTask {
    pub fn to_chat_messages(&self) -> Result<Vec<ChatMessage>> {
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
        backfill_tool_result_names(&mut messages);
        let tools = choose_tools(Some(&tools), self.tool_choice.as_ref())
            .map_err(|_| anyhow!("Invalid benchmark tool_choice"))?;
        insert_tools_message(&mut messages, &tools);
        Ok(messages)
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
