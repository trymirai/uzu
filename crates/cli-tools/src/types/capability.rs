use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Grammar,
    Tracing,
    #[serde(rename = "cli")]
    CLI,
}

impl Capability {
    pub fn feature(&self) -> String {
        match self {
            Capability::Grammar => "capability-grammar".to_string(),
            Capability::Tracing => "capability-tracing".to_string(),
            Capability::CLI => "capability-cli".to_string(),
        }
    }
}
