use serde::{Deserialize, Serialize};

use crate::types::Bindings;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct LanguageConfig {
    pub targets: Vec<String>,
    pub tools: Vec<String>,
    pub bindings: Vec<Bindings>,
}
