use clap::ValueEnum;
use heck::{ToLowerCamelCase, ToSnakeCase};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Case {
    Snake,
    LowerCamel,
    Kebab,
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

    pub fn case(&self) -> Case {
        match self {
            Language::Rust | Language::Python => Case::Snake,
            Language::TypeScript => Case::LowerCamel,
            Language::Swift => Case::Kebab,
        }
    }

    pub fn convert_name(
        &self,
        name: &str,
    ) -> String {
        match self.case() {
            Case::Snake => name.to_snake_case(),
            Case::LowerCamel => name.to_lower_camel_case(),
            Case::Kebab => name.to_string(),
        }
    }
}
