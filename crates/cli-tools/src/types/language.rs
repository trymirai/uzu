use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Language {
    Rust,
    Python,
    Swift,
    #[serde(rename = "typescript")]
    #[value(name = "typescript")]
    TypeScript,
}

impl Language {
    pub fn name(&self) -> String {
        match self {
            Language::Rust => "rust".to_string(),
            Language::Python => "python".to_string(),
            Language::Swift => "swift".to_string(),
            Language::TypeScript => "typescript".to_string(),
        }
    }
}
