use clap::ValueEnum;
use heck::{ToLowerCamelCase, ToSnakeCase, ToUpperCamelCase};
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
    UpperCamel,
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

    pub fn display(&self) -> String {
        match self {
            Language::Rust => "Rust".to_string(),
            Language::Python => "Python".to_string(),
            Language::Swift => "Swift".to_string(),
            Language::TypeScript => "TypeScript".to_string(),
        }
    }

    pub fn code_fence(&self) -> &'static str {
        match self {
            Language::Rust => "rust",
            Language::Python => "python",
            Language::Swift => "swift",
            Language::TypeScript => "ts",
        }
    }

    pub fn command_argument_case(&self) -> Case {
        match self {
            Language::Rust | Language::Python => Case::Snake,
            Language::TypeScript => Case::LowerCamel,
            Language::Swift => Case::Kebab,
        }
    }

    pub fn file_case(&self) -> Case {
        match self {
            Language::Swift => Case::UpperCamel,
            _ => self.command_argument_case(),
        }
    }

    pub fn file_extension(&self) -> &'static str {
        match self {
            Language::Rust => "rs",
            Language::Python => "py",
            Language::Swift => "swift",
            Language::TypeScript => "ts",
        }
    }

    pub fn convert_command_name(
        &self,
        name: &str,
    ) -> String {
        format(name, self.command_argument_case())
    }

    pub fn convert_file_name(
        &self,
        name: &str,
    ) -> String {
        format(name, self.file_case())
    }
}

fn format(
    name: &str,
    case: Case,
) -> String {
    match case {
        Case::Snake => name.to_snake_case(),
        Case::LowerCamel => name.to_lower_camel_case(),
        Case::UpperCamel => name.to_upper_camel_case(),
        Case::Kebab => name.to_string(),
    }
}
