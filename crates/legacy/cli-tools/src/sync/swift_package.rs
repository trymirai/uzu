use anyhow::{Context, Result};

use crate::{
    configs::{PlatformsConfig, WorkspaceManifest},
    sync::{SyncTask, replace::replace_required},
};

pub struct SwiftPackageSyncTask;

impl SyncTask for SwiftPackageSyncTask {
    fn process(
        &self,
        platforms: &PlatformsConfig,
        _workspace: &WorkspaceManifest,
        input: &str,
    ) -> Result<String> {
        let ios = platforms
            .envs
            .get("IPHONEOS_DEPLOYMENT_TARGET")
            .context("Missing IPHONEOS_DEPLOYMENT_TARGET in platforms.toml [envs]")?;
        let macos = platforms
            .envs
            .get("MACOSX_DEPLOYMENT_TARGET")
            .context("Missing MACOSX_DEPLOYMENT_TARGET in platforms.toml [envs]")?;

        let output = replace_required(input, r#"\.iOS\("[^"]*"\)"#, format!(r#".iOS("{ios}")"#))?;
        let output = replace_required(&output, r#"\.macCatalyst\("[^"]*"\)"#, format!(r#".macCatalyst("{ios}")"#))?;
        replace_required(&output, r#"\.macOS\("[^"]*"\)"#, format!(r#".macOS("{macos}")"#))
    }
}
