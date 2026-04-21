use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::session::classification::Stats;

#[bindings::export(Struct, name = "ClassificationOutput")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Output {
    pub logits: Vec<f64>,
    pub probabilities: HashMap<String, f64>,
    pub stats: Stats,
}
