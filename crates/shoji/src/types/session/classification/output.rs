use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::session::classification::ClassificationStats;

#[bindings::export(Struct)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassificationOutput {
    pub logits: Vec<f64>,
    pub probabilities: HashMap<String, f64>,
    pub stats: ClassificationStats,
}
