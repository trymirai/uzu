use serde::{Deserialize, Serialize};

#[bindings::export(Struct)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcmBatch {
    pub samples: Vec<f64>,
    pub sample_rate: u32,
    pub channels: u32,
    pub lengths: Vec<u32>,
}
