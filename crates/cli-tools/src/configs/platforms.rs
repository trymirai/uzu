use std::{fs, path::PathBuf};

use anyhow::{Context, Ok, Result, anyhow};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::{
    configs::{LanguageConfig, Paths, TargetConfig, ToolConfig},
    types::{Backend, Bindings, Capability, Language, Tool},
};

pub const HOST_TARGET: &str = "host";
pub const ALL_TARGET: &str = "all";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ExampleConfig {
    pub title: String,
    pub description: String,
    pub explanation: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct PlatformsConfig {
    pub envs: IndexMap<String, String>,
    pub badges: IndexMap<String, String>,
    pub examples: IndexMap<String, ExampleConfig>,
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

    pub fn host_target(&self) -> Result<String> {
        let arch = std::env::consts::ARCH;
        let os = std::env::consts::OS;
        let name = match os {
            "macos" => format!("{arch}-apple-darwin"),
            "linux" => format!("{arch}-unknown-linux-gnu"),
            "windows" => format!("{arch}-pc-windows-msvc"),
            _ => return Err(anyhow!("Unsupported host OS: {os}")),
        };
        Ok(name)
    }

    pub fn targets_for_language(
        &self,
        language: Language,
        requested_targets: Vec<String>,
    ) -> Result<Vec<String>> {
        let host_target = self.host_target()?;
        let target_configs = self.targets.iter().collect::<Vec<_>>();
        let requested_targets = target_configs
            .iter()
            .filter(|(name, config)| {
                requested_targets.contains(name)
                    || config.aliases.iter().any(|alias| requested_targets.contains(alias))
                    || (requested_targets.contains(&HOST_TARGET.to_string()) && (*name == &host_target))
                    || requested_targets.contains(&ALL_TARGET.to_string())
            })
            .map(|(name, _)| (*name).clone())
            .collect::<Vec<_>>();

        let language_config = self.languages.get(&language).context("Language not found")?;
        let language_targets = language_config.targets.clone();

        let resolved_targets = language_targets
            .iter()
            .filter(|target| requested_targets.contains(target))
            .map(|target| target.clone())
            .collect::<Vec<_>>();
        Ok(resolved_targets)
    }

    pub fn backend_for_target(
        &self,
        target: String,
    ) -> Result<Backend> {
        let target_config = self.targets.get(&target).context("Target not found")?;
        Ok(target_config.backend.clone())
    }

    pub fn capabilities_for_target(
        &self,
        target: String,
        requested_capabilities: Vec<Capability>,
    ) -> Result<Vec<Capability>> {
        let target_config = self.targets.get(&target).context("Target not found")?;

        let default_capabilities = target_config.capabilities_default.clone();
        if requested_capabilities.is_empty() {
            return Ok(default_capabilities);
        }

        let supported_capabilities = target_config.capabilities_supported.clone();
        Ok(requested_capabilities
            .iter()
            .filter(|capability| supported_capabilities.contains(capability))
            .map(|capability| capability.clone())
            .collect::<Vec<_>>())
    }

    pub fn bindings_for_language(
        &self,
        language: Language,
    ) -> Result<Vec<Bindings>> {
        let language_config = self.languages.get(&language).context("Language not found")?;
        Ok(language_config.bindings.clone())
    }

    pub fn examples_path_for_language(
        &self,
        language: Language,
    ) -> Result<PathBuf> {
        let language_config = self.languages.get(&language).context("Language not found")?;
        Ok(Paths::new()?.root_path.join(&language_config.examples_path))
    }

    pub fn required_envs_for_target(
        &self,
        target: String,
    ) -> Result<IndexMap<String, String>> {
        let target_config = self.targets.get(&target).context("Target not found")?;
        let required_envs = target_config.required_envs.clone();
        let mut envs = IndexMap::new();
        for env in required_envs {
            envs.insert(env.clone(), self.envs.get(&env).context("Environment variable not found")?.clone());
        }
        Ok(envs)
    }
}
