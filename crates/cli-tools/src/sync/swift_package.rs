use anyhow::{Context, Result};
use regex::Regex;

use crate::{
    configs::{PlatformsConfig, WorkspaceManifest},
    sync::SyncTask,
};

pub struct SwiftPackageSyncTask;

impl SyncTask for SwiftPackageSyncTask {
    fn process(
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

        let mut output = input.to_string();
        output = Regex::new(r#"\.iOS\("[^"]*"\)"#)?.replace(&output, format!(r#".iOS("{ios}")"#).as_str()).into_owned();
        output = Regex::new(r#"\.macOS\("[^"]*"\)"#)?
            .replace(&output, format!(r#".macOS("{macos}")"#).as_str())
            .into_owned();
        Ok(output)
    }
}
