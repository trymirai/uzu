use std::fs;

use anyhow::{Context, Result};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::{
    configs::{LanguageConfig, Paths, TargetConfig, ToolConfig},
    types::{Language, Tool},
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct PlatformsConfig {
    pub tools: IndexMap<String, ToolConfig>,
    pub targets: IndexMap<String, TargetConfig>,
    pub languages: IndexMap<Language, LanguageConfig>,
}

impl PlatformsConfig {
    pub fn load() -> Result<Self> {
        let path = Paths::new()?.platforms_toml();
        let body = fs::read_to_string(&path)?;
        Ok(toml::from_str(&body)?)
    }

    pub fn tools_for_language(
        &self,
        language: Language,
    ) -> Result<Vec<Tool>> {
        let tool_names = self.languages.get(&language).context("Language not found")?.tools.clone();
        let tools: Vec<_> = tool_names
            .iter()
            .map(|name| {
                self.tools.get(name).context("Tool not found").map(|config| Tool {
                    name: name.clone(),
                    provider: config.provider.clone(),
                    version: config.version.clone(),
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(tools)
    }
}
