use serde::{Deserialize, Serialize};

#[bindings::export(Struct)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Quantization {
    pub method: String,
    pub bits_per_weight: u32,
}
