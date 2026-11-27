use anyhow::{Context, Result};
use serde::Serialize;

use crate::{
    configs::{PlatformsConfig, WorkspaceManifest},
    sync::SyncTask,
};

/// Writes the app's macOS Tauri config, which Tauri merges over `tauri.conf.json`, so the minimum macOS version
/// follows platforms.toml.
pub struct MiraiGuiSyncTask;

#[derive(Serialize)]
struct TauriConfig<'a> {
    bundle: Bundle<'a>,
}

#[derive(Serialize)]
struct Bundle<'a> {
    #[serde(rename = "macOS")]
    mac_os: MacOsBundle<'a>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct MacOsBundle<'a> {
    minimum_system_version: &'a str,
}

impl SyncTask for MiraiGuiSyncTask {
    fn process(
        &self,
        platforms: &PlatformsConfig,
        _workspace: &WorkspaceManifest,
        _input: &str,
    ) -> Result<String> {
        let minimum_system_version = platforms
            .envs
            .get("MACOSX_DEPLOYMENT_TARGET")
            .context("Missing MACOSX_DEPLOYMENT_TARGET in platforms.toml [envs]")?;
        let config = TauriConfig {
            bundle: Bundle {
                mac_os: MacOsBundle {
                    minimum_system_version,
                },
            },
        };
        Ok(serde_json::to_string_pretty(&config)? + "\n")
    }
}
