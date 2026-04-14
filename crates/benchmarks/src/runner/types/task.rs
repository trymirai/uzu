use serde::{Deserialize, Serialize};
use uzu::session::types::Message;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum BenchRunMode {
    #[default]
    WarmedProcess,
    FreshSession,
    FreshProcess,
}

fn default_warmup_tokens() -> u64 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub identifier: String,
    pub repo_id: String,
    pub number_of_runs: u64,
    pub tokens_limit: u64,
    pub messages: Vec<Message>,
    pub greedy: bool,
    #[serde(default)]
    pub forced_token_path: Option<Box<[u64]>>,
    #[serde(default)]
    pub reference_text_blake3: Option<String>,
    #[serde(default)]
    pub run_mode: BenchRunMode,
    #[serde(default = "default_warmup_tokens")]
    pub warmup_tokens: u64,
}
