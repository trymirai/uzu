use anyhow::{Context, Result};
use toml_edit::{DocumentMut, Item, Value};

use crate::{
    configs::{PlatformsConfig, WorkspaceManifest},
    sync::SyncTask,
};

pub struct ToolchainsSyncTask;

impl SyncTask for ToolchainsSyncTask {
    fn process(
        platforms: &PlatformsConfig,
        _workspace: &WorkspaceManifest,
        input: &str,
    ) -> Result<String> {
        let mut document: DocumentMut = input.parse()?;
        let targets = document
            .get_mut("toolchain")
            .and_then(Item::as_table_mut)
            .and_then(|table| table.get_mut("targets"))
            .and_then(Item::as_array_mut)
            .context("Missing toolchain.targets in rust-toolchain.toml")?;

        targets.clear();
        for name in platforms.targets.keys() {
            let mut value = Value::from(name.as_str());
            value.decor_mut().set_prefix("\n    ");
            targets.push_formatted(value);
        }
        targets.set_trailing("\n");
        targets.set_trailing_comma(true);

        Ok(document.to_string())
    }
}
