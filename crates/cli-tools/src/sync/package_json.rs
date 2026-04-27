use std::collections::BTreeSet;

use anyhow::{Context, Result};
use regex::Regex;

use crate::{
    configs::{PlatformsConfig, WorkspaceManifest},
    sync::SyncTask,
    types::Language,
};

pub struct PackageJsonSyncTask;

impl SyncTask for PackageJsonSyncTask {
    fn process(
        platforms: &PlatformsConfig,
        workspace: &WorkspaceManifest,
        input: &str,
    ) -> Result<String> {
        let package = &workspace.workspace.package;

        let mut output = input.to_string();
        output = replace_string_field(&output, "version", &package.version)?;
        output = replace_string_field(&output, "description", &package.description)?;
        output = replace_string_field(&output, "author", &package.authors.join(", "))?;
        output = replace_string_field(&output, "repository", &package.repository)?;
        output = replace_string_field(&output, "license", &package.license)?;

        let typescript_targets = platforms
            .languages
            .get(&Language::TypeScript)
            .context("Missing [languages.typescript] in platforms.toml")?
            .targets
            .clone();

        let cpu = unique(typescript_targets.iter().map(|target| target_cpu(target)).collect::<Result<Vec<_>>>()?);
        let os = unique(typescript_targets.iter().map(|target| target_os(target)).collect::<Result<Vec<_>>>()?);

        output = replace_string_array_field(&output, "cpu", cpu)?;
        output = replace_string_array_field(&output, "os", os)?;

        Ok(output)
    }
}

fn target_cpu(target: &str) -> Result<String> {
    if target.starts_with("aarch64-") {
        Ok("arm64".to_string())
    } else if target.starts_with("x86_64-") {
        Ok("x64".to_string())
    } else {
        Err(anyhow::anyhow!("Unsupported target arch: {target}"))
    }
}

fn target_os(target: &str) -> Result<String> {
    if target.contains("-apple-darwin") {
        Ok("darwin".to_string())
    } else if target.contains("-unknown-linux") {
        Ok("linux".to_string())
    } else if target.contains("-pc-windows") {
        Ok("win32".to_string())
    } else {
        Err(anyhow::anyhow!("Unsupported target os: {target}"))
    }
}

fn unique(values: Vec<String>) -> Vec<String> {
    BTreeSet::from_iter(values).into_iter().collect()
}

fn replace_string_field(
    input: &str,
    key: &str,
    value: &str,
) -> Result<String> {
    let pattern = format!(r#"("{key}"\s*:\s*)"[^"]*""#);
    let regex = Regex::new(&pattern)?;
    let literal = serde_json::to_string(value)?;
    let replacement = format!("$1{literal}");
    Ok(regex.replace(input, replacement.as_str()).into_owned())
}

fn replace_string_array_field(
    input: &str,
    key: &str,
    values: Vec<String>,
) -> Result<String> {
    let pattern = format!(r#"("{key}"\s*:\s*)\[[^\]]*\]"#);
    let regex = Regex::new(&pattern)?;
    let items = values.iter().map(|value| serde_json::to_string(value)).collect::<Result<Vec<_>, _>>()?;
    let replacement = format!("$1[\n        {}\n    ]", items.join(",\n        "));
    Ok(regex.replace(input, replacement.as_str()).into_owned())
}
