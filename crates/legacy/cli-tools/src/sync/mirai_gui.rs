use anyhow::{Context, Result};
use regex::Regex;
use toml_edit::{DocumentMut, value};

use crate::{
    configs::{PlatformsConfig, WorkspaceManifest},
    sync::SyncTask,
};

pub enum MiraiGuiSyncTask {
    PackageJson,
    CargoToml,
    TauriConf,
}

impl SyncTask for MiraiGuiSyncTask {
    fn process(
        &self,
        platforms: &PlatformsConfig,
        workspace: &WorkspaceManifest,
        input: &str,
    ) -> Result<String> {
        let version = &workspace.workspace.package.version;
        match self {
            Self::PackageJson => replace_string_field(input, "version", version),
            Self::CargoToml => {
                let mut document: DocumentMut = input.parse()?;
                document["package"]["version"] = value(version);
                Ok(document.to_string())
            },
            Self::TauriConf => {
                let target = platforms
                    .envs
                    .get("MACOSX_DEPLOYMENT_TARGET")
                    .context("Missing MACOSX_DEPLOYMENT_TARGET in platforms.toml [envs]")?;
                replace_string_field(input, "minimumSystemVersion", target)
            },
        }
    }
}

fn replace_string_field(
    input: &str,
    key: &str,
    value: &str,
) -> Result<String> {
    let regex = Regex::new(&format!(r#"("{key}"\s*:\s*)"[^"]*""#))?;
    let literal = serde_json::to_string(value)?;
    Ok(regex.replace(input, format!("$1{literal}").as_str()).into_owned())
}
