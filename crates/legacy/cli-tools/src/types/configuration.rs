use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Configuration {
    Debug,
    Release,
}

impl Configuration {
    pub fn name(&self) -> String {
        match self {
            Configuration::Debug => "debug".to_string(),
            Configuration::Release => "release".to_string(),
        }
    }
}
