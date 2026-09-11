use serde::{Deserialize, Serialize};

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct SamplingParameters {
    pub temperature: Option<f64>,
    pub top_k: Option<i64>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub suffix_repetition_length: Option<i64>,
}
