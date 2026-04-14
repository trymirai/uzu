use serde::{Deserialize, Serialize};
use uzu::ConfigDataType;

use crate::runner::types::{Device, Task};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Result {
    pub task: Task,
    pub device: Device,
    pub engine_version: String,
    pub timestamp: u64,
    pub precision: Option<ConfigDataType>,
    pub memory_used: Option<u64>,
    pub kv_storage_bytes: u64,
    pub tokens_count_input: u64,
    pub tokens_count_output: u64,
    pub time_to_first_token: f64,
    pub prompt_tokens_per_second: f64,
    pub generate_tokens_per_second: Option<f64>,
    pub text_blake3: String,
    pub matches_reference_output: Option<bool>,
    pub text: String,
}
