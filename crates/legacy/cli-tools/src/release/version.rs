use std::fs;

use anyhow::{Context, Result};
use toml_edit::{DocumentMut, value};

use crate::configs::Paths;

pub fn bump_workspace_version(version: &str) -> Result<()> {
    let path = Paths::new()?.workspace_manifest_path();
    let body = fs::read_to_string(&path).with_context(|| format!("Failed to read {}", path.display()))?;
    let mut document: DocumentMut = body.parse()?;
    document["workspace"]["package"]["version"] = value(version);
    fs::write(&path, document.to_string())?;
    Ok(())
}
