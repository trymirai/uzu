use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Grammar,
    Tracing,
}

impl Capability {
    pub fn feature(&self) -> String {
        match self {
            Capability::Grammar => "capability-grammar".to_string(),
            Capability::Tracing => "capability-tracing".to_string(),
        }
    }
}
