use serde::{Deserialize, Serialize};

#[bindings::export(Struct)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TotalStats {
    pub duration: f64,
    pub tokens_count_input: u32,
    pub tokens_count_output: u32,
}
