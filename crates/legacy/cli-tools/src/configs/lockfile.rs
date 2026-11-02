use std::fs;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::configs::Paths;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LockedPackage {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CargoLock {
    pub package: Vec<LockedPackage>,
}

impl CargoLock {
    pub fn load() -> Result<Self> {
        let path = Paths::new()?.cargo_lock_path();
        let body = fs::read_to_string(&path).with_context(|| format!("Failed to read {}", path.display()))?;
        let lock: CargoLock = toml::from_str(&body)?;
        Ok(lock)
    }

    pub fn package_version(
        &self,
        name: &str,
    ) -> Result<String> {
        let versions = self
            .package
            .iter()
            .filter(|package| package.name == name)
            .map(|package| package.version.clone())
            .collect::<Vec<_>>();
        match versions.as_slice() {
            [version] => Ok(version.clone()),
            [] => Err(anyhow!("Package `{name}` not found in Cargo.lock")),
            _ => Err(anyhow!("Multiple versions of `{name}` in Cargo.lock: {versions:?}")),
        }
    }
}
