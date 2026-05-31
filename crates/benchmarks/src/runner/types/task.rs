use backend_uzu::session::types::Message;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SpeculatorSpec {
    PromptLookup {
        max_ngram_size: usize,
        number_of_speculated_tokens: usize,
    },
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
    pub speculator: Option<SpeculatorSpec>,
}
