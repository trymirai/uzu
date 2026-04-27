use anyhow::{Context, Result};
use regex::Regex;

use crate::{
    configs::{PlatformsConfig, WorkspaceManifest},
    sync::SyncTask,
};

pub struct PlaygroundSyncTask;

impl SyncTask for PlaygroundSyncTask {
    fn process(
        &self,
        platforms: &PlatformsConfig,
        workspace: &WorkspaceManifest,
        input: &str,
    ) -> Result<String> {
        let package = &workspace.workspace.package;
        let ios_deployment_target = platforms
            .envs
            .get("IPHONEOS_DEPLOYMENT_TARGET")
            .context("Missing IPHONEOS_DEPLOYMENT_TARGET in platforms.toml [envs]")?;
        let macos_deployment_target = platforms
            .envs
            .get("MACOSX_DEPLOYMENT_TARGET")
            .context("Missing MACOSX_DEPLOYMENT_TARGET in platforms.toml [envs]")?;

        let mut output = input.to_string();

        let marketing_version = Regex::new(r#"("MARKETING_VERSION"\s*:\s*)"[^"]*""#)?;
        output = marketing_version.replace(&output, format!(r#"$1"{}""#, package.version).as_str()).into_owned();

        let multiplatform = Regex::new(r#"\.multiplatform\(\s*iOS:\s*"[^"]*"\s*,\s*macOS:\s*"[^"]*"\s*\)"#)?;
        output = multiplatform
            .replace(
                &output,
                format!(r#".multiplatform(iOS: "{ios_deployment_target}", macOS: "{macos_deployment_target}")"#)
                    .as_str(),
            )
            .into_owned();

        Ok(output)
    }
}
