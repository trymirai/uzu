use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
    pub tool_calls: Option<Vec<Value>>,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchSampling {
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub temp: Option<f32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchRequest {
    pub prompt_text: Option<String>,
    pub prompt_chat: Option<Vec<ChatMessage>>,
    pub tools: Option<Vec<Value>>,
    pub tool_choice: Option<Value>,

    pub max_tokens: Option<usize>,
    pub speculative_depth: Option<usize>,
    pub sampling: Option<BenchSampling>,
    pub num_runs: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResponse {
    pub text: String,
    pub time_to_first_token: f64,
    pub prompt_tps: f64,
    pub decode_tps: f64,
    pub tokens_per_forward_pass: f64,
    pub duration: f64,
    pub memory_phys_footprint: u64,
    pub memory_resident: u64,
    pub memory_graphics_total: u64,
}
