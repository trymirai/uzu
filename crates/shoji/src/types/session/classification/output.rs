use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::session::classification::ClassificationStats;

#[bindings::export(Struct)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassificationOutputProbabilities {
    pub values: HashMap<String, f64>,
}

#[bindings::export(ClassCloneable)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassificationOutput {
    pub logits: Vec<f64>,
    pub probabilities: ClassificationOutputProbabilities,
    pub stats: ClassificationStats,
}
