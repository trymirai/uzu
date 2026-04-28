use serde::{Deserialize, Serialize};

use crate::{configs::ToolProvider, types::Command};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Tool {
    pub name: String,
    pub provider: ToolProvider,
    pub version: String,
}

impl Tool {
    pub fn command(&self) -> Command {
        match self.provider {
            ToolProvider::Cargo => Command::cargo_install(&self.name, &self.version),
            ToolProvider::Uv => Command::uv_install(&self.name, &self.version),
        }
    }
}
