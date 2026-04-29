use serde::{Deserialize, Serialize};

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassificationStats {
    pub preprocessing_duration: f64,
    pub forward_pass_duration: f64,
    pub postprocessing_duration: f64,
    pub total_duration: f64,
    pub tokens_count: i64,
    pub tokens_per_second: f64,
    pub predicted_label: String,
    pub confidence: f64,
}
