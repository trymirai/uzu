use serde::{Deserialize, Serialize};
use uzu::session::types::Message;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub identifier: String,
    pub repo_id: String,
    pub number_of_runs: u64,
    pub tokens_limit: u64,
    pub messages: Vec<Message>,
}
