use std::fs;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::configs::Paths;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspacePackage {
    pub version: String,
    pub description: String,
    pub authors: Vec<String>,
    pub homepage: String,
    pub repository: String,
    pub license: String,
    pub readme: String,
    #[serde(default)]
    pub keywords: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceConfig {
    pub package: WorkspacePackage,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceManifest {
    pub workspace: WorkspaceConfig,
}

impl WorkspaceManifest {
    pub fn load() -> Result<Self> {
        let path = Paths::new()?.workspace_manifest_path();
        let body = fs::read_to_string(&path).with_context(|| format!("Failed to read {}", path.display()))?;
        let manifest: WorkspaceManifest = toml::from_str(&body)?;
        Ok(manifest)
    }
}
