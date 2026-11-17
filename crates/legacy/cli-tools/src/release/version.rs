use std::fs;

use anyhow::{Context, Result};
use toml_edit::{DocumentMut, value};

use crate::configs::Paths;

pub fn bump_workspace_version(version: &str) -> Result<()> {
    edit_workspace_manifest(|document| document["workspace"]["package"]["version"] = value(version))
}

pub fn bump_published_version(version: &str) -> Result<()> {
    edit_workspace_manifest(|document| document["workspace"]["metadata"]["published"]["version"] = value(version))
}

fn edit_workspace_manifest(edit: impl FnOnce(&mut DocumentMut)) -> Result<()> {
    let path = Paths::new()?.workspace_manifest_path();
    let body = fs::read_to_string(&path).with_context(|| format!("Failed to read {}", path.display()))?;
    let mut document: DocumentMut = body.parse()?;
    edit(&mut document);
    fs::write(&path, document.to_string())?;
    Ok(())
}
