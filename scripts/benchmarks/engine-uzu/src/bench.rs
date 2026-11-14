use serde::{Deserialize, Serialize};
use serde_json::Value;
use uzu::types::basic::ReasoningEffort;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
    pub tool_calls: Option<Vec<Value>>,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GreedySampling {
    pub greedy: monostate::MustBe!(true),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StochasticSampling {
    #[serde(deserialize_with = "Option::deserialize")]
    pub top_k: Option<i32>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub top_p: Option<f32>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub min_p: Option<f32>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub temp: Option<f32>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub repetition_penalty: Option<f32>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub suffix_repetition_length: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BenchSampling {
    Greedy(GreedySampling),
    Stochastic(StochasticSampling),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Option<Vec<Value>>,
    pub tool_choice: Option<Value>,

    pub max_output_tokens: Option<usize>,
    pub reasoning_effort: ReasoningEffort,
    pub seed: Option<u32>,
}

/// For each effort a caller may want, the effort to request from this model, which it supports.
#[expect(dead_code, reason = "callers apply the mapping before they send a request")]
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReasoningEfforts {
    pub disabled: ReasoningEffort,
    pub default: ReasoningEffort,
    pub low: ReasoningEffort,
    pub medium: ReasoningEffort,
    pub high: ReasoningEffort,
    pub xhigh: ReasoningEffort,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CooldownRequest {
    #[serde(default = "default_cooldown_temperature")]
    pub temperature: f64,
    #[serde(default)]
    pub only_if_capped: bool,
}

fn default_cooldown_temperature() -> f64 {
    60.0
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum Request {
    BenchRequest(BenchRequest),
    CooldownRequest(CooldownRequest),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FinishReason {
    Stop,
    Length,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResponse {
    pub text: String,
    pub finish_reason: FinishReason,
    pub input_tokens_count: usize,
    pub output_tokens_count: usize,
    pub time_to_first_token: f64,
    pub prefill_tps: f64,
    pub decode_tps: f64,
    pub tokens_per_forward_pass: Option<f64>,
    pub duration: f64,
    pub energy_gpu: f64,
    pub memory_used: u64,
    pub memory_phys_footprint: u64,
    pub memory_resident: u64,
    pub memory_graphics_total: u64,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum Response {
    BenchResponse(BenchResponse),
    ReadyResponse,
    CooldownResponse,
}
