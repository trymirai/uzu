use serde::{Deserialize, Serialize};

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassificationLabel {
    pub index: i64,
    pub label: String,
    pub title: String,
    pub description: String,
    pub recommended_threshold: f64,
}
