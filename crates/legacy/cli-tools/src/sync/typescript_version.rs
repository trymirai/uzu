use anyhow::{Result, anyhow};
use regex::Regex;

use crate::{
    configs::{PlatformsConfig, WorkspaceManifest},
    sync::SyncTask,
};

/// The files the TypeScript build stamps with the version, kept in step without building.
pub enum TypeScriptVersionSyncTask {
    Constant,
    NapiLoader,
}

impl SyncTask for TypeScriptVersionSyncTask {
    fn process(
        &self,
        _platforms: &PlatformsConfig,
        workspace: &WorkspaceManifest,
        input: &str,
    ) -> Result<String> {
        let version = &workspace.workspace.package.version;
        match self {
            Self::Constant => replace_versions(input, r"(export const VERSION = ')[^']*(')", version),
            Self::NapiLoader => {
                let output = replace_versions(input, r"(bindingPackageVersion !== ')[^']*(')", version)?;
                replace_versions(&output, r"(version mismatch, expected )\S+( but got)", version)
            },
        }
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
