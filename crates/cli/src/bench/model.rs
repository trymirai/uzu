use backend_uzu::data_type::DataType;
use rocket::serde::{Deserialize, Serialize};
use uzu::types::session::chat::{ChatMessage, ChatRole};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchTask {
    pub identifier: String,
    pub repo_id: String,
    pub number_of_runs: u64,
    pub tokens_limit: u64,
    pub messages: Vec<BenchMessage>,
    pub greedy: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BenchMessage {
    pub role: BenchMessageRole,
    pub content: String,
    pub reasoning_content: Option<String>,
}

impl BenchMessage {
    pub fn to_chat_message(&self) -> ChatMessage {
        let role = match self.role {
            BenchMessageRole::Assistant => ChatRole::Assistant {},
            BenchMessageRole::System => ChatRole::System {},
            BenchMessageRole::User => ChatRole::User {},
        };

        let mut message = ChatMessage::for_role(role).with_text(self.content.clone());
        if let Some(reasoning) = &self.reasoning_content {
            message = message.with_reasoning(reasoning.clone())
        };

        message
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum BenchMessageRole {
    System,
    User,
    Assistant,
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
    pub text: String,
}
