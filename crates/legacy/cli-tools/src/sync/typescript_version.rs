use anyhow::{Result, anyhow};
use regex::Regex;

use crate::{
    configs::{PlatformsConfig, WorkspaceManifest},
    sync::SyncTask,
};

/// The napi loader the TypeScript build stamps with the version, kept in step without building.
pub struct TypeScriptVersionSyncTask;

impl SyncTask for TypeScriptVersionSyncTask {
    fn process(
        &self,
        _platforms: &PlatformsConfig,
        workspace: &WorkspaceManifest,
        input: &str,
    ) -> Result<String> {
        let version = &workspace.workspace.package.version;
        let output = replace_versions(input, r"(bindingPackageVersion !== ')[^']*(')", version)?;
        replace_versions(&output, r"(version mismatch, expected )\S+( but got)", version)
    }
}

fn replace_versions(
    input: &str,
    pattern: &str,
    version: &str,
) -> Result<String> {
    let regex = Regex::new(pattern)?;
    if !regex.is_match(input) {
        return Err(anyhow!("Nothing matches {pattern}"));
    }
    Ok(regex.replace_all(input, format!("${{1}}{version}${{2}}").as_str()).into_owned())
}
