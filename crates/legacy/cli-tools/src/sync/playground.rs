use anyhow::{Context, Result};

use crate::{
    configs::{PlatformsConfig, WorkspaceManifest},
    sync::{SyncTask, replace::replace_required},
};

pub struct PlaygroundSyncTask;

impl SyncTask for PlaygroundSyncTask {
    fn process(
        &self,
        platforms: &PlatformsConfig,
        workspace: &WorkspaceManifest,
        input: &str,
    ) -> Result<String> {
        let ios_deployment_target = platforms
            .envs
            .get("IPHONEOS_DEPLOYMENT_TARGET")
            .context("Missing IPHONEOS_DEPLOYMENT_TARGET in platforms.toml [envs]")?;
        let macos_deployment_target = platforms
            .envs
            .get("MACOSX_DEPLOYMENT_TARGET")
            .context("Missing MACOSX_DEPLOYMENT_TARGET in platforms.toml [envs]")?;

        let output = replace_required(
            input,
            r#"("MARKETING_VERSION"\s*:\s*)"[^"]*""#,
            format!(r#"${{1}}"{}""#, workspace.workspace.package.version),
        )?;
        replace_required(
            &output,
            r#"\.multiplatform\(\s*iOS:\s*"[^"]*"\s*,\s*macOS:\s*"[^"]*"\s*\)"#,
            format!(r#".multiplatform(iOS: "{ios_deployment_target}", macOS: "{macos_deployment_target}")"#),
        )
    }
}
