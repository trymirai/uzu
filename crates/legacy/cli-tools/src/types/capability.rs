use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Grammar,
}

impl Capability {
    pub fn feature(&self) -> String {
        match self {
            Capability::Grammar => "capability-grammar".to_string(),
        }
    }
}
